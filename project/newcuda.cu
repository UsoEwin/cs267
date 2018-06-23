#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <stdio.h>
#include <getopt.h>
#include <iostream>
#include <string>
#include <omp.h>
#include <cuda.h>

#define ITEM_SIZE 3
#define EMPTY                 0b10011000 //ENABLE_bit, Black_bit, White_bit, Black_liberty_bit, White_liberty_bit, NC, NC, remove_temp_bit
#define ENABLE_bit            0b10000000
#define White_bit             0b00100000
#define Black_bit             0b01000000
#define Black_liberty_bit     0b00010000
#define White_liberty_bit     0b00001000
#define remove_temp_bit       0b10000001
#define full_mask             0b0000000011111111
#define BOARD_SIZE 19
#define EB 256
#define INVALID -1
#define FIND_COLOR(Arr, index) EB*Arr[index+1]+Arr[index+2]
#define FIND_INDEX(X, Y, BZ) ITEM_SIZE*((BZ+2)*(Y)+(X))
#define FIND_STATUS(Arr, index, bit) (Arr[index])&bit


using namespace std;

typedef set<unsigned short> COLOR_SET;
typedef map<unsigned short, COLOR_SET> COLOR_MAP;

struct Board{
	char board[(BOARD_SIZE+2)*(BOARD_SIZE+2)*ITEM_SIZE];
	COLOR_MAP color_map_liberty; //DO HERE
	unsigned short color_counter;
};


void construct_board(Board* go_board){
	go_board->color_counter = 0;
	#pragma omp parallel for
	for(int i = 0; i < ITEM_SIZE*(BOARD_SIZE+2)*(BOARD_SIZE+2); i++) {
		go_board->board[i] = 0;
	}
	#pragma omp parallel for
	for(int i = 1; i <= BOARD_SIZE; i++) {
		#pragma omp parallel for
		for(int j = 1; j <= BOARD_SIZE; j++) {
			go_board->board[ITEM_SIZE*((BOARD_SIZE+2)*i+j)] = EMPTY;
		}
	}
}

void print_board_stone(Board* go_board) {
	for(int i = 1; i <= BOARD_SIZE; i++) {
		for(int j = 1; j <= BOARD_SIZE; j++) {
			if (go_board->board[FIND_INDEX(j,i,BOARD_SIZE)] & White_bit) {
				printf(" o");
			} else if (go_board->board[FIND_INDEX(j,i,BOARD_SIZE)]& Black_bit) {
				printf(" x");
			} else {
				printf(" .");
			}
		}
		printf("\n");
	}
}

void print_board_layer(Board* go_board, char layer_bit, int shift_right) {
	for(int i = 1; i <= BOARD_SIZE; i++) {
		for(int j = 1; j <= BOARD_SIZE; j++) {
			printf("%4d", ((go_board->board[FIND_INDEX(j,i,BOARD_SIZE)]&layer_bit)>>shift_right)& 0b00000001);
		}
		printf("\n");
	}
}

void print_board_info(Board* go_board) {
	cout << "Color Map Size = " << go_board->color_map_liberty.size()<< "\nHave:\n";
	for (COLOR_MAP::iterator i = go_board->color_map_liberty.begin(); i!=go_board->color_map_liberty.end(); ++i) {
		cout << "KEY = " << i->first << ": {";
		for (COLOR_SET::iterator j = i->second.begin(); j!=i->second.end(); ++j) {
			cout << *j << ", ";
		}
		cout << "}\n";
	}
	std::cout << '\n';
}

__global__ void parallel_reduce_add_kernel_2D(char* d_in, unsigned short* d_out){
	int index = FIND_INDEX(threadIdx.x+1, blockIdx.x+1, blockDim.x);
	int tid = threadIdx.x;
	extern __shared__ char sdata1[];
    sdata1[tid] = (d_in[index]);
    __syncthreads();

	for(unsigned int i = 16 ; i > 0; i >>=1){
		if(tid < i){
			sdata1[tid] += sdata1[tid + i];
		}
		__syncthreads();
	}
	if (tid == 0){
		d_out[blockIdx.x] = sdata1[0];
	}
}

__global__ void parallel_reduce_add_kernel_Black_2D(char* d_in, unsigned short* d_out){
	int index = FIND_INDEX(threadIdx.x+1, blockIdx.x+1, blockDim.x);
	int tid = threadIdx.x;
	extern __shared__ char sdata2[];
    sdata2[tid] = (d_in[index] & Black_liberty_bit) >> 4;
    __syncthreads();

	for(unsigned int i = 16 ; i > 0; i >>=1){
		if(tid < i){
			sdata2[tid] += sdata2[tid + i];
		}
		__syncthreads();
	}
	if (tid == 0){
		d_out[blockIdx.x] = sdata2[0];
		printf("From BLACK CUDA[%d]: %d\n", blockIdx.x, d_out[blockIdx.x]);
	}
}

__global__ void parallel_reduce_add_kernel_White_2D(char* d_in, unsigned short* d_out){
	int index = FIND_INDEX(threadIdx.x+1, blockIdx.x+1, blockDim.x);
	int tid = threadIdx.x;
	extern __shared__ char sdata3[];
    sdata3[tid] = (d_in[index] & White_liberty_bit) >> 3;
    __syncthreads();

	for(unsigned int i = 16 ; i > 0; i >>=1){
		if(tid < i){
			sdata3[tid] += sdata3[tid + i];
			printf("From WHITE CUDA[index=%d][tid=%d]:=%d,%d\n", index, tid, sdata3[tid],sdata3[tid + i]);
		}
		__syncthreads();
	}
	if (tid == 0){
		d_out[blockIdx.x] = sdata3[0];
	}
}

__global__ void parallel_reduce_add_kernel_1D(unsigned short* d_in, unsigned short* d_out) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	extern __shared__ unsigned short mem[];
    mem[tid] = (d_in[index]);
    __syncthreads();

	for(unsigned int i = 16 ; i > 0; i >>=1){
		if(tid < i){
			mem[tid] += mem[tid + i];
		}
		__syncthreads();
	}
	if (tid == 0){
		d_out[blockIdx.x] = mem[0];
	}
}

/*__global__ void f(char* d_out, char* d_in) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
}*/
__global__ void color_merge(char* d_board, unsigned short* color_array) {
	int index = FIND_INDEX(threadIdx.x+1, blockIdx.x+1, blockDim.x);
	unsigned short color =  color_array[0];
	unsigned short current_color = FIND_COLOR(d_board, index);
	if ((current_color  ==  color_array[1]  ||
		 current_color  ==  color_array[2]  ||
		 current_color  ==  color_array[3]) && current_color != 0) {
		d_board[index+1] = (color >> 8) & full_mask;
		d_board[index+2] = color & full_mask;
	}
}

__global__ void clear_stone(char* d_board, unsigned short* color_remove_array, unsigned short* d_flag) {
	int index = FIND_INDEX(threadIdx.x+1, blockIdx.x+1, blockDim.x);
	unsigned short current_color = FIND_COLOR(d_board, index);
	if ((current_color == color_remove_array[0] || 
		 current_color == color_remove_array[1] || 
		 current_color == color_remove_array[2] || 
		 current_color == color_remove_array[3]) && current_color != 0 && d_flag[0]) {

		d_board[index] = remove_temp_bit | (d_board[index] ^ 0b00011000);
		d_board[index+1] = 0;
		d_board[index+2] = 0;
	}
}

__global__ void check_stone_self(char* d_board, char* d_self_counter, unsigned short* color_remove_array) {
	int index = FIND_INDEX(threadIdx.x+1, blockIdx.x+1, blockDim.x);

	int left_index = FIND_INDEX(threadIdx.x, blockIdx.x+1, blockDim.x);
	int right_index = FIND_INDEX(threadIdx.x+2, blockIdx.x+1, blockDim.x);
	int up_index = FIND_INDEX(threadIdx.x+1, blockIdx.x, blockDim.x);
	int down_index = FIND_INDEX(threadIdx.x+1, blockIdx.x+2, blockDim.x);

	unsigned short current_color = FIND_COLOR(d_board, index);

	if ((current_color == color_remove_array[0]) && (current_color != 0) && ((d_board[left_index] & remove_temp_bit)||
																			 (d_board[right_index] & remove_temp_bit)||
																			 (d_board[up_index] & remove_temp_bit)||
																			 (d_board[down_index] & remove_temp_bit)) ) {
		d_self_counter[index] = 1;
	} else {
		d_self_counter[index] = 0;
	}
}

//[1<=x<=19] [1<=y<=19]
unsigned int check_color(Board* go_board, char ifBlack, char x, char y) {
	unsigned short center = FIND_INDEX(x,y, BOARD_SIZE);
	if (!(go_board->board[center] & ENABLE_bit)) {
		return INVALID;
	}
	for (int i = 1; i<= 19; i++) {
		for (int j = 1; j <= 19 ; j++) {
			cout << EB*go_board->board[FIND_INDEX(j, i, BOARD_SIZE)+1]+ go_board->board[FIND_INDEX(j, i, BOARD_SIZE)+2] <<" ";
		}
		cout << "\n";
	}

	int i = 0;
	int j = 0;
	int k;
	unsigned int return_value;
	char* d_board;
	unsigned short* d_color_array_self;
	unsigned short* d_color_array_rival;
	unsigned short* d_score_black_arr;
	unsigned short* d_score_white_arr;
	unsigned short* d_score_black;
	unsigned short* d_score_white;
	char* d_self_counter;
	unsigned short* score_black = new unsigned short[1];
	unsigned short* score_white = new unsigned short[1];
	int d_board_size = (BOARD_SIZE+2)*(BOARD_SIZE+2)*ITEM_SIZE;
	// update center position status: enable+Black_bit+White_bit+liberty
	go_board->board[center] = 0b00000000 | (White_bit<<ifBlack) | (White_liberty_bit<<ifBlack);
	// find local status around center position
	unsigned short up = FIND_INDEX(x, y-1, BOARD_SIZE);
	unsigned short down = FIND_INDEX(x, y+1, BOARD_SIZE);
	unsigned short right = FIND_INDEX(x+1, y, BOARD_SIZE);
	unsigned short left = FIND_INDEX(x-1, y, BOARD_SIZE);

	unsigned short up_color = FIND_COLOR(go_board->board, up);
	unsigned short down_color = FIND_COLOR(go_board->board, down);
	unsigned short right_color = FIND_COLOR(go_board->board, right);
	unsigned short left_color = FIND_COLOR(go_board->board, left);

	COLOR_SET color_set_empty_index;
	unsigned short* color_array_self = new unsigned short[4];
	unsigned short* color_array_rival = new unsigned short[4];
	// 
	if (up_color) {
		if (FIND_STATUS(go_board->board, up, White_bit << ifBlack))
			color_array_self[i++] = up_color;
		if (FIND_STATUS(go_board->board, up, Black_bit >> ifBlack))
			color_array_rival[j++] = up_color;
	} else if (FIND_STATUS(go_board->board, up, ENABLE_bit)) {
		color_set_empty_index.insert(up);
	}
	if (right_color) {
		if (FIND_STATUS(go_board->board, right, White_bit << ifBlack))
			color_array_self[i++] = right_color;
		if (FIND_STATUS(go_board->board, right, Black_bit >> ifBlack))
			color_array_rival[j++] = right_color;
	} else if (FIND_STATUS(go_board->board, right, ENABLE_bit)) {
		color_set_empty_index.insert(right);
	}
	if (down_color) {
		if (FIND_STATUS(go_board->board, down, White_bit << ifBlack))
			color_array_self[i++] = down_color;
		if (FIND_STATUS(go_board->board, down, Black_bit >> ifBlack))
			color_array_rival[j++] = down_color;
	} else if (FIND_STATUS(go_board->board, down, ENABLE_bit)) {
		color_set_empty_index.insert(down);
	}
	if (left_color) {
		if (FIND_STATUS(go_board->board, left, White_bit << ifBlack))
			color_array_self[i++] = left_color;
		if (FIND_STATUS(go_board->board, left, Black_bit >> ifBlack))
			color_array_rival[j++] = left_color;
	} else if (FIND_STATUS(go_board->board, left, ENABLE_bit)) {
		color_set_empty_index.insert(left);
	}
	cout << "i = " << i << ", j = " << j <<"\n";

	cudaMalloc((void **)&(d_board), d_board_size*sizeof(char));
	cudaMalloc((void **)&(d_self_counter), d_board_size*sizeof(char));
	cudaMalloc((void **)&(d_color_array_self), 4*sizeof(unsigned short));
	cudaMalloc((void **)&(d_color_array_rival), 4*sizeof(unsigned short));
	cudaMalloc((void **)&(d_score_black_arr), (BOARD_SIZE+2)*sizeof(unsigned short));
	cudaMalloc((void **)&(d_score_white_arr), (BOARD_SIZE+2)*sizeof(unsigned short));
	cudaMalloc((void **)&(d_score_black), sizeof(unsigned short));
	cudaMalloc((void **)&(d_score_white), sizeof(unsigned short));
	
	//TO DO HERE
	if (i > 0) {
		unsigned short target_color_value = color_array_self[0];
		//merge color to color_array_self[0]
		for (int ii = 1; ii < i; ii++) {
			set_union(go_board->color_map_liberty.find(target_color_value)->second.begin(),
					  go_board->color_map_liberty.find(target_color_value)->second.end(),
					  go_board->color_map_liberty.find(color_array_self[ii])->second.begin(),
					  go_board->color_map_liberty.find(color_array_self[ii])->second.end(),
					  inserter(go_board->color_map_liberty.find(target_color_value)->second,
							   go_board->color_map_liberty.find(target_color_value)->second.begin()));
			go_board->color_map_liberty.erase(color_array_self[ii]);
		}
		set_union(go_board->color_map_liberty.find(target_color_value)->second.begin(),
				  go_board->color_map_liberty.find(target_color_value)->second.end(),
				  color_set_empty_index.begin(), color_set_empty_index.end(),
				  inserter(go_board->color_map_liberty.find(target_color_value)->second,
						   go_board->color_map_liberty.find(target_color_value)->second.begin()));
		go_board->color_map_liberty.find(target_color_value)->second.erase(center);
		//update center position color on CPU
		go_board->board[center+1] = (target_color_value >> 8) & full_mask;
		go_board->board[center+2] = target_color_value & full_mask;
		//
		cudaMemcpy(d_color_array_self, color_array_self, 4*sizeof(unsigned short), cudaMemcpyHostToDevice);
		cudaMemcpy(d_board, go_board->board, d_board_size*sizeof(char), cudaMemcpyHostToDevice);
		color_merge<<<BOARD_SIZE, BOARD_SIZE>>>(d_board, d_color_array_self);
	} else {
		//create new color
		cout << "i = " << i << ", j = " << j <<"\n";
		unsigned short color_value = ++go_board->color_counter;
		go_board->color_map_liberty[color_value] = color_set_empty_index;
		//update center position color on CPU
		go_board->board[center+1] = (color_value >> 8) & full_mask;
		go_board->board[center+2] = color_value & full_mask;
		//
		color_array_self[i++] = color_value;
		cout << "i = " << i << ", j = " << j <<"\n";
		cudaMemcpy(d_color_array_self, color_array_self, 4*sizeof(unsigned short), cudaMemcpyHostToDevice);
		cudaMemcpy(d_board, go_board->board, d_board_size*sizeof(char), cudaMemcpyHostToDevice);
	}
	// remove center from rival color_array_rival
	k = j;
	#pragma omp parallel for
	for (int jj = 0; jj < j; jj++) {
		go_board->color_map_liberty.find(color_array_rival[jj])->second.erase(center);
		if (go_board->color_map_liberty.find(color_array_rival[jj])->second.size()) {
			color_array_rival[jj] = 0;
			k--;
		} else {
			go_board->color_map_liberty.erase(color_array_rival[jj]);
		}
	}
	if (k > 0) {
		score_black[0] = 1;
		cudaMemcpy(d_score_black, score_black, sizeof(unsigned short), cudaMemcpyHostToDevice);
		cudaMemcpy(d_color_array_rival, color_array_rival, 4*sizeof(unsigned short), cudaMemcpyHostToDevice);
		clear_stone<<<BOARD_SIZE, BOARD_SIZE>>>(d_board, d_color_array_rival, d_score_black);
	}
	printf("CUDA1\n");
	if (go_board->color_map_liberty.find(color_array_self[0])->second.size() == 0){
		printf("CUDA1.1\n");
		cudaMemcpy(d_color_array_self, color_array_self, sizeof(unsigned short), cudaMemcpyHostToDevice);
		check_stone_self<<<BOARD_SIZE, BOARD_SIZE>>>(d_board, d_self_counter, d_color_array_self);
		printf("CUDA1.2\n");
		parallel_reduce_add_kernel_2D<<<BOARD_SIZE, BOARD_SIZE, (32)*sizeof(char)>>>(d_self_counter, d_score_black_arr);
		parallel_reduce_add_kernel_1D<<<1, BOARD_SIZE, (32)*sizeof(unsigned short)>>>(d_score_black_arr, d_score_black);
		printf("CUDA1.3\n");
		clear_stone<<<BOARD_SIZE, BOARD_SIZE>>>(d_board, d_color_array_self, d_score_black);
	}
	printf("CUDA2\n");
	parallel_reduce_add_kernel_White_2D<<<BOARD_SIZE, BOARD_SIZE, (32)*sizeof(char)>>>(d_board, d_score_white_arr);
	parallel_reduce_add_kernel_Black_2D<<<BOARD_SIZE, BOARD_SIZE, (32)*sizeof(char)>>>(d_board, d_score_black_arr);
	printf("CUDA2.1\n");
	parallel_reduce_add_kernel_1D<<<1, BOARD_SIZE, (32)*sizeof(unsigned short)>>>(d_score_white_arr, d_score_white);
	parallel_reduce_add_kernel_1D<<<1, BOARD_SIZE, (32)*sizeof(unsigned short)>>>(d_score_black_arr, d_score_black);
	// reduction for score calculation
	printf("CUDA3\n");
	cudaMemcpy(go_board->board, d_board, d_board_size*sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(score_black, d_score_black, sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cudaMemcpy(score_white, d_score_white, sizeof(unsigned short), cudaMemcpyDeviceToHost);
	return_value = (((score_black[0])<<16)| (score_white[0]));
	printf("BALCK: %4d, WHITE: %4d\n", score_black, score_white);
	// finished GPU acceleration.
	// start 
	// #pragma omp parallel for 
	// for (int i2 = 1; i2 <= BOARD_SIZE; ++i2) {
	// 	#pragma omp parallel for
	// 	for (int j2 = 0; j2 <= BOARD_SIZE; ++j2) {
	// 		unsigned char center_index = FIND_INDEX(j2, i2, BOARD_SIZE);
	// 	}
	// }
	cudaFree(d_board);
	cudaFree(d_color_array_self);
	cudaFree(d_color_array_rival);
	cudaFree(d_score_black_arr);
	cudaFree(d_score_white_arr);
	cudaFree(d_score_white);
	cudaFree(d_score_black);
	cudaFree(d_self_counter);
	delete[]	color_array_rival;
	delete[]	color_array_self;
	delete[]	score_white;
	delete[]	score_black;
	return return_value;
}

int main() {
	Board* GoBoard = new Board();
	construct_board(GoBoard);

	GoBoard->board[FIND_INDEX(2,1,BOARD_SIZE)] = 0b01010000;
	GoBoard->board[FIND_INDEX(2,1,BOARD_SIZE)+2] = 0b00000001;
	GoBoard->board[FIND_INDEX(1,2,BOARD_SIZE)] = 0b01010000;
	GoBoard->board[FIND_INDEX(1,2,BOARD_SIZE)+2] = 0b00000010;
	GoBoard->board[FIND_INDEX(3,2,BOARD_SIZE)] = 0b01010000;
	GoBoard->board[FIND_INDEX(3,2,BOARD_SIZE)+2] = 0b00000011;
	GoBoard->board[FIND_INDEX(2,3,BOARD_SIZE)] = 0b01010000;
	GoBoard->board[FIND_INDEX(2,3,BOARD_SIZE)+2] = 0b00000100;
	GoBoard->board[FIND_INDEX(2,2,BOARD_SIZE)] = 0b10010000;
	GoBoard->board[FIND_INDEX(1,1,BOARD_SIZE)] = 0b10010000;

	COLOR_SET fk1;
	fk1.insert(FIND_INDEX(1,1,BOARD_SIZE));
	fk1.insert(FIND_INDEX(3,1,BOARD_SIZE));
	fk1.insert(FIND_INDEX(2,2,BOARD_SIZE));
	COLOR_SET fk2;
	fk2.insert(FIND_INDEX(1,1,BOARD_SIZE));
	fk2.insert(FIND_INDEX(2,2,BOARD_SIZE));
	fk2.insert(FIND_INDEX(1,3,BOARD_SIZE));
	COLOR_SET fk3;
	fk3.insert(FIND_INDEX(3,1,BOARD_SIZE));
	fk3.insert(FIND_INDEX(2,2,BOARD_SIZE));
	fk3.insert(FIND_INDEX(3,3,BOARD_SIZE));
	fk3.insert(FIND_INDEX(4,2,BOARD_SIZE));
	COLOR_SET fk4;
	fk4.insert(FIND_INDEX(2,2,BOARD_SIZE));
	fk4.insert(FIND_INDEX(2,4,BOARD_SIZE));
	fk4.insert(FIND_INDEX(1,3,BOARD_SIZE));
	fk4.insert(FIND_INDEX(3,3,BOARD_SIZE));
	GoBoard->color_map_liberty[1] = fk1;
	GoBoard->color_map_liberty[2] = fk2;
	GoBoard->color_map_liberty[3] = fk3;
	GoBoard->color_map_liberty[4] = fk4;
	print_board_stone(GoBoard);
	cout << "ENABLE_bit:\n";
	print_board_layer(GoBoard, ENABLE_bit, 7);
	cout << "Black_bit:\n";
	print_board_layer(GoBoard, Black_bit, 6);
	cout << "White_bit:\n";
	print_board_layer(GoBoard, White_bit, 5);
	cout << "Black_liberty_bit:\n";
	print_board_layer(GoBoard, Black_liberty_bit, 4);
	cout << "White_liberty_bit:\n";
	print_board_layer(GoBoard, White_liberty_bit, 3);
	print_board_info(GoBoard);

	unsigned int score = check_color(GoBoard, 1, 2, 2);
	
	print_board_stone(GoBoard);
	cout << "ENABLE_bit:\n";
	print_board_layer(GoBoard, ENABLE_bit, 7);
	cout << "Black_bit:\n";
	print_board_layer(GoBoard, Black_bit, 6);
	cout << "White_bit:\n";
	print_board_layer(GoBoard, White_bit, 5);
	cout << "Black_liberty_bit:\n";
	print_board_layer(GoBoard, Black_liberty_bit, 4);
	cout << "White_liberty_bit:\n";
	print_board_layer(GoBoard, White_liberty_bit, 3);
	print_board_info(GoBoard);

	delete GoBoard;
	return 0;
}
