#include "go.h"

inline int raisePwr(int num, int times){
	int pwr = 1;
	for (int i = 0; i < times; ++i)
		pwr *= num;
	return pwr;
}


//kernel function, search depth n
int serialkernelMonteCarlo(GameBoard* myboard, int n){
	int size = myboard->size;
	int search = myboard->size;
	if (n == 2 && size == 19) search = 8;
	if (n == 3 && size == 9) search = 5;
	if (n == 3 && size == 19) search = 4;
	int pwr = raisePwr(search, n<<1);
	int partial_pwr = int(pwr/search*search);

	int xcor = 0;
	int ycor = 0;
	int prevrow = myboard->last_move/size;
	int prevcol = myboard->last_move%size;

	//check boundary
	if (prevrow + search>>1 >= size) xcor = size - search;
	else if (prevrow - search>>1 > 0) xcor = prevrow - search>>1;

	if (prevcol + search>>1 >= size) ycor = size - search;
	else if (prevcol - search>>1 > 0) ycor = prevcol - search>>1;

	//generate random starting pt
	int maxpos = rand() %(size*size);
	float maxval = -101.0;

	int nextstep;
	for (int ii=xcor; ii<xcor + search; ii++){
		for (int jj=ycor; jj<ycor + search; jj++){
			nextstep = ii * size + jj;
			if (myboard->draw[nextstep] == 0){
				int local_cnt = 0;
				int local_sum = 0;
				for (int p = 0; p < partial_pwr; p++){

					GameBoard* next_board = new GameBoard;
					buildBoard(next_board, size);
					
					//rebuild the board
					for (int i=0; i<size; i++){
						
						for (int j=0; j<size; j++){
							if (myboard->draw[i*size+j] != 0){
								if (myboard->draw[i*size+j] == 1){
									addStone(next_board, i, j, 1);
								} else {
									addStone(next_board, i, j, -1);
								}
							}
						}
					}

					int flag = addStone(next_board, nextstep / size, nextstep % size, -1);
					int next_next;
					int type = -1;// color
					if (flag == 1){
						for (int k=0; k<n-1; k++){
							next_next = rand() % (size * size);
							flag = addStone(next_board, next_next / size, next_next % size, type);
							type *= (-1);
							if (flag == 0) break;
						}
					}
					if (flag != 0){
						getTerr(next_board);
						int w_count = 0;
						for (int i=0; i<size; i++){
							for (int j=0; j<size; j++){
								if (next_board->classify[i*size+j] == 1){
									w_count -= 1;
								} else if (next_board->classify[i*size+j] == -1){
									w_count += 1;
								}
							}
						}
						local_cnt += 1;
						local_sum += w_count;
					}
					delete next_board;
				}
				if (float(local_sum) / local_cnt > maxval){
            		maxval = float(local_sum) / local_cnt;
            		maxpos = nextstep;
        		}
			}
		}
	}

	return maxpos;
}

