#include "board.h"

//stones
#define WHITE (-1)
#define BLACK (1)
//operators
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define abs(a) ((a)>=(0)?(a):(-a))

int checkStone(GameBoard* myboard, int row, int col, int state);
void getTerr(GameBoard* this_board);
//functions for compute MCTS distance
inline int map(int dist){
	switch(dist) {
		case 4 : return 1;
		case 3 : return 2;
		case 2 : return 4;
		case 1 : return 8;
		case 0 : return 16;
		return 0;
	}
}

inline void cleanBoard(GameBoard* myboard){
	for (int i = 0; i < 361; ++i)
		myboard->visited[i] = 0;
	return;
}

void buildBoard(GameBoard* myboard, int size){
	myboard->size = size;
	//for a new game, always let black play first
	myboard->current_player_state = BLACK;
	myboard->last_move = WHITE;
	for (int i = 0; i < size; ++i){
		for (int j = 0; j < size; ++j){	
			//initialize if this a stone or not
			myboard->draw[i*size+j] = 0;
			myboard->eval[i*size+j] = 0;
			myboard->classify[i*size+j] = 0;
		}
	}
	return;
}

//utility function to print the board
void printBoard(GameBoard* myboard){
	for (int i = 0; i < myboard->size; ++i){
		for (int j = 0; j < myboard->size; ++j){
			if (myboard->draw[i*myboard->size+j]){
				if (myboard->draw[i*myboard->size+j] == BLACK)
					printf(" x");
				else printf(" o"); //WHITE
			}
			else printf(" .");
		}
		printf("\n");
	}
	for (int i = 0; i < myboard->size; ++i){
		printf(" $");
	}
	printf("\n");
	return;
}

//add one stone to the game
int addStone(GameBoard* myboard, int row, int col, int state){
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
	myboard->last_move = row*size+col;
	getTerr(myboard);
	//for test purpose
	//printClasearchify(myboard);
	return 1;
}

//remove stones from the game
void deleteStone(GameBoard* myboard, int row, int col){
	//recursively traverse
	int size = myboard->size;
	if (myboard->visited[row*size+col])
		return;
	myboard->visited[row*size+col] = 1;
	int state = myboard->visited[row*size+col];
	//make sure not hit the boundary
	if (row > 0){	
		if (myboard->draw[(row-1)*size+col] == state) {
			deleteStone(myboard, row-1, col);
			myboard->draw[(row-1)*size+col] = 0;
		}
	}	
	if(row < size - 1){
		if(myboard->draw[(row+1)*size+col] == state){
			deleteStone(myboard, row+1, col);
			myboard->draw[(row+1)*size+col] = 0;
		}
	}
	if (col>0){
		if (myboard->draw[row*size+col-1] == state){
			deleteStone(myboard, row, col-1);
			myboard->draw[row*size+col-1] = 0;
		}
	}
	if (col<size-1){
		if (myboard->draw[row*size+col+1] == state){
			deleteStone(myboard, row, col+1);
			myboard->draw[row*size+col+1] = 0;
		}
	}
	myboard->draw[row*size+col] = 0;
	return;
}

int countLiberty(GameBoard* myboard, int row, int col){
	int size = myboard->size;
	int count = 0;
	//check boundary
	if (myboard->visited[row*size+col] == 1)
		return 0;
	myboard->visited[row*size+col] = 1;
	int state = myboard->draw[row*size+col];
	if (row > 0){	
		if (myboard->draw[(row-1)*size+col] == state) 
			count += countLiberty(myboard,row-1,col);
		//same color or no opponent stone
		else count += (myboard->draw[(row-1)*size+col] == 0);
	}
	if (row < size-1){	
		if (myboard->draw[(row+1)*size+col] == state) 
			count += countLiberty(myboard,row+1,col);
		//same color or no opponent stone
		else count += (myboard->draw[(row+1)*size+col] == 0);
	}
	if (col > 0){	
		if (myboard->draw[row*size+col-1] == state) 
			count += countLiberty(myboard,row,col-1);
		//same color or no opponent stone
		else count += (myboard->draw[row*size+col-1] == 0);
	}
	if (col < size-1){	
		if (myboard->draw[row*size+col+1] == state) 
			count += countLiberty(myboard,row,col+1);
		//same color or no opponent stone
		else count += (myboard->draw[row*size+col+1] == 0);
	}

	return count;
}
//see if this stone is dead
int checkStone(GameBoard* myboard, int row, int col, int state){
	int neighbors[4];
	int size = myboard->size;
	//set the boundary
	if (row > 0) neighbors[0] = (row-1)*size +col; 
	else neighbors[0] = -1;
	if (row < size-1) neighbors[1] = (row+1)*size +col; 
	else neighbors[1] = -1;
	if (col > 0) neighbors[2] = row*size +col-1; 
	else neighbors[2] = -1;	 
	if (col < size-1) neighbors[3] = row*size +col+1; 
	else neighbors[3] = -1;	 

	int flag = 1;
	//dead
	if (!countLiberty(myboard, row, col))
		flag = 0;
	int indr,indc;
	for (int i = 0; i < 4; ++i)
	{
		if (neighbors[i] != -1 && myboard->draw[neighbors[i]] == -state){
			indr = neighbors[i]/size;
			indc = neighbors[i]%size;
			cleanBoard(myboard);
			//dead
			if (!countLiberty(myboard, indr, indc)){
				cleanBoard(myboard);
				deleteStone(myboard, indr, indc);
				flag = 1;
			}
		}
	}
	return flag;
}

static inline int raisePwr(int num, int times){
	int pwr = 1;
	for (int i = 0; i < times; ++i)
		pwr *= num;
	return pwr;
}

//kernel function, search depth n
int kernelMonteCarlo(GameBoard* myboard, int n){
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

void getTerr(GameBoard* myboard) {
	int size = myboard->size;

	for (int i=0; i< size*size; i++){
		myboard->eval[i] = 0;
	}

	int idx, dist, diff;
	for (int r = 0; r < size; r++){
		for (int c = 0; c < size; c++){
			idx = r * size + c;
			if (myboard->draw[idx] != 0){
				diff = myboard->draw[idx];
				for(int i = max(r - 4, 0); i < min(r + 5, size); i++){
					for(int j = max(c - 4, 0); j < min(c + 5, size); j++){
						dist = abs(r - i) + abs(c - j);
						myboard->eval[i * size + j] += diff * map(dist);
					}
				}
			}
		}
	}

	for(int i = 0; i < size * size; i++) {
    	if(myboard->draw[i] == 1) {
      		if(myboard->eval[i] <= 0) myboard->classify[i] = -1; //dead
      		else myboard->classify[i] = 1;
    	}
    	else if(myboard->draw[i] == -1) {
      		if(myboard->eval[i] >= 0) myboard->classify[i] = 1; //dead
      		else myboard->classify[i] = -1;
    	}
    	else if(myboard->eval[i] > 0) myboard->classify[i] = 1;
    	else if(myboard->eval[i] < 0) myboard->classify[i] = -1;
    	else myboard->classify[i] = 0;
  	}
}
/*

void board_printclassify(GameBoard* myboard){
	int s = myboard->size;
	for (int i = 0; i < s; i++){
		for (int j = 0; j < s; j++){
			if (myboard->classify[i*s+j] != 0){
				if (myboard->classify[i*s+j] == 1){
					printf(" B");
				} else {
					printf(" W");
				}
			} else {
				printf(" .");
			}
		}
		printf("\n");
	}
	for (int i=0; i<100; i++){
		printf("#");
	}
	printf("\n");
}
*/