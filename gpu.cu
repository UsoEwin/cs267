#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "go.h"

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
	//myboard->last_move = row*size+col;
	//getTerr(myboard);
	//for test purpose
	//printClasearchify(myboard);
	return 1;
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

    cudaMalloc(&device_stones, num * s * s * sizeof(int));
    cudaMemcpy(device_stones, stones, num * s * s * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&device_result, num * sizeof(int));
    cudaMemcpy(device_result, result, num * sizeof(int), cudaMemcpyHostToDevice);


    kernel_monte_carlo<<<blocks, threadsPerBlock>>>(device_stones, s, device_result);
    cudaThreadSynchronize();

    cudaMemcpy(result, device_result, num * sizeof(int), cudaMemcpyDeviceToHost);

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
