#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "go.h"

#include <chrono>
using namespace std;
using namespace std::chrono;

static inline int cudaraisePwr(int num, int times){
	int pwr = 1;
	for (int i = 0; i < times; ++i)
		pwr *= num;
	return pwr;
}

int cudaaddStone(GameBoard* myboard, int row, int col, int state){
	//check boundary
	if (row < 0 || row >= myboard->size || col < 0 || col >= myboard->size )
		return 0;
	int size = myboard->size;
	//stone exist
	if(myboard->draw[row*size+col]) return 0; 
	//first put the stone, then check the state
	myboard->draw[row*size+col] = state;

	if (!checkStone(myboard, row, col, state)){
		//get removed
		myboard->draw[row*size+col] = 0;
		return 0;
	}
	return 1;
}

__global__ void
kernel_monte_carlo(int* stones, int s, int* result){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (stones[index * s * s] != -2){
        int eval[361];
        for (int i=0; i< s*s; i++){
            eval[i] = 0;
        }

        int idx, dist, diff;
        //calculate eval
        for (int r = 0; r < s; r++){
            for (int c = 0; c < s; c++){
                idx = r * s + c;
                if (stones[idx + index * s * s] != 0){
                    diff = stones[idx + index * s * s];

                    int i1,i2,j1,j2;

                    if (r-4 > 0){i1 = r-4;} else {i1=0;}
                    if (r+5 < s){i2 = r+5;} else {i2=s;}
                    if (c-4 > 0){j1 = c-4;} else {j1=0;}
                    if (c+5 < s){j2 = c+5;} else {j2=s;}

                    for(int i = i1; i < i2; i++){
                        for(int j = j1; j < j2; j++){

                            int ab1, ab2,m;

                            if (r-i > 0){ ab1 = r-i;}
                            else {ab1 = i-r;}
                            if (c-j > 0){ ab2 = c-j;}
                            else {ab2 = j-c;}
                            dist = ab1+ab2;
                            switch(dist) {
                            case 4 : m = 1; break;
                            case 3 : m = 2; break;
                            case 2 : m = 4; break;
                            case 1 : m = 8; break;
                            case 0 : m = 16; break;
                            default: m = 0; break;

                            }
                            eval[i * s + j] += diff * m;
                        }
                    }
                }
            }
        }

        int w_count = 0;
        for(int i = 0; i < s * s; i++) {
            if(stones[i + index * s * s] == 1) {
                if(eval[i] < 0) w_count += 1;
                else w_count -= 1;
            }
            else if(stones[i + index * s * s] == -1) {
                if(eval[i] > 0) w_count -= 1;
                else w_count += 1;
            }
            else if(eval[i] > 0) w_count -= 1;
            else if(eval[i] < 0) w_count += 1;
        }
        result[index] = w_count;
    }
}


int cudaMonteCarlo(GameBoard* this_board, int n) {
    int s = this_board->size;
    int ss = s;
    if (n == 2 and s == 19) ss = 8;
    if (n == 3 and s == 9) ss = 5;
    if (n == 3 and s == 19) ss = 4;
    int num = cudaraisePwr(ss, 2*n);
    int partial_num = int(num / (ss * ss));

    const int threadsPerBlock = 128;
    const int blocks = (num + threadsPerBlock - 1) / threadsPerBlock;

    int result[num];
    for(int i = 0; i < num; i++) result[i] = 100;

    int stones[num * s * s];
    int move_seq[num * n];

    //generating moving sequences

    int startx = 0;
    int starty = 0;
    int last_row = this_board->last_move / s;
    int last_col = this_board->last_move % s;
    if (last_row + int(ss / 2) >= s){startx = s - ss;}
    else if (last_row - int(ss / 2) > 0) {startx = last_row - int(ss/2);}

    if (last_col + int(ss / 2) >= s){starty = s - ss;}
    else if (last_col - int(ss / 2) > 0) {starty = last_col - int(ss/2);}

    int p = 0;
    //printf("startx = %d, starty = %d\n", startx, starty);
    for (int i=startx; i<startx + ss; i++){
        for (int j=starty; j<starty + ss; j++){
            for (int k=1; k<partial_num; k++){
                move_seq[p * n] = i * s + j;
                p += 1;    
            }
        }
    }

    for (int i=0; i<num * n; i++){
        if (i % n != 0) move_seq[i] = rand() % (s * s);
    }

    for (int idx = 0; idx < num; idx ++){
        GameBoard* next_board = new GameBoard;
        buildBoard(next_board, s);
        for (int r = 0; r < s; r++){
            for (int c = 0; c < s; c++){
                next_board->draw[r * s + c] = this_board->draw[r * s + c];
            }
        }

        int flag = 1;
        int type = 1;
        int cur_flag;
        for (int k=0; k<n; k++){
            type *= (-1);
            cur_flag = cudaaddStone(next_board, move_seq[idx * n + k] / s, move_seq[idx * n + k] % s, -1);
            if (cur_flag == 0){
                flag = 0;
                break;
            }
        }

        if (flag == 1){
            for (int i = 0; i < s * s; i++){
                stones[idx * s * s + i] = next_board->draw[i];
            }
        } else {
            stones[idx * s * s] = -2;
        }
        delete next_board;
    }

    int* device_stones;
    int* device_result; 
    
    //for timing purpose
    high_resolution_clock::time_point mem_s = high_resolution_clock::now();  
    cudaMalloc(&device_stones, num * s * s * sizeof(int));
    cudaMemcpy(device_stones, stones, num * s * s * sizeof(int), cudaMemcpyHostToDevice);




    cudaMalloc(&device_result, num * sizeof(int));
    cudaMemcpy(device_result, result, num * sizeof(int), cudaMemcpyHostToDevice);

    high_resolution_clock::time_point mem_e = high_resolution_clock::now();
    duration<double> mem = duration_cast<duration<double>>(mem_e - mem_s);  


    high_resolution_clock::time_point kernel_s = high_resolution_clock::now(); 

    kernel_monte_carlo<<<blocks, threadsPerBlock>>>(device_stones, s, device_result);

    high_resolution_clock::time_point kernel_e = high_resolution_clock::now(); 

    duration<double> kernel = duration_cast<duration<double>>(kernel_e - kernel_s);
	

    high_resolution_clock::time_point synch_s = high_resolution_clock::now();
    cudaThreadSynchronize();
    high_resolution_clock::time_point synch_e = high_resolution_clock::now();
    duration<double> kernel = duration_cast<duration<double>>(synch_e - synch_s);


    cudaMemcpy(result, device_result, num * sizeof(int), cudaMemcpyDeviceToHost);


    cout<<"memcpy time : " << mem.count() <<"sec"<< endl;
    cout<<"kernel time : " << kernel.count() <<"sec"<< endl;
    cout<<"synch time : " << synch.count() <<"sec"<< endl;
    //can be parallized by omp or whatever
    int max_pos = rand() % (s * s);
    float max_val = -101.0;
    int local_sum = 0;
    int local_cnt = 0;
    for (int idx = 0; idx < ss * ss; idx++){
        local_sum = 0;
        local_cnt = 0;
        for (int i=0; i < partial_num; i++){
            if (result[idx * partial_num + i] != 100){
                local_cnt += 1;
                local_sum += result[idx * partial_num + i];
            }
        }

        if (float(local_sum) / local_cnt > max_val){
            max_val = float(local_sum) / local_cnt;
            max_pos = move_seq[idx * partial_num];
        }
    }
    cudaFree(result);
    cudaFree(device_stones);
    cudaFree(device_result);

    return max_pos;
}
