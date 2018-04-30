#include "readfile.h"

void clear_visited(GameBoard* this_board);
void board_printboard(GameBoard* this_board);
void delete_stone(GameBoard* this_board, int row, int col);
int get_liberties(GameBoard* this_board, int row, int col);
int checkStone(GameBoard* this_board, int row, int col, int state);
void board_get_terr(GameBoard* this_board);
void board_printclassify(GameBoard* this_board);
int board_monte_carlo(GameBoard* this_board);

int min_of(int x, int y) {
  if(x < y) return x;
  return y;
}

int max_of(int x, int y) {
  if(x > y) return x;
  return y;
}

int abso(int x) {
  if(x < 0) return -x;
  return x;
}

int mapping(int dist) {
  if(dist == 4) return 1;
  if(dist == 3) return 2;
  if(dist == 2) return 4;
  if(dist == 1) return 8;
  if(dist == 0) return 16;
  return 0;
}

void clear_visited(GameBoard* this_board){
	for (int i=0; i<361; i++){
		this_board->visited[i] = 0;
	}
}

void board_construct(GameBoard* this_board, int s){
	this_board->size = s;
	this_board->current_player_state = 1;
	this_board->last_move = -1;
	for (int i=0; i<s; i++){
		for (int j=0; j<s; j++){
			this_board->draw[i*s+j] = 0;
			this_board->eval[i*s+j] = 0;
			this_board->classify[i*s+j] = 0;
		}
	}
}

void board_printboard(GameBoard* this_board){
	int s = this_board->size;
	for (int i = 0; i < s; i++){
		// printf("#%d",i+1);
		for (int j = 0; j < s; j++){
			if (this_board->draw[i*s+j] != 0){
				if (this_board->draw[i*s+j] == 1){
					printf(" x");
				} else {
					printf(" o");
				}
			} else {
				printf(" .");
			}
		}
		printf("\n");
	}
	for (int i=0; i<40; i++){
		printf("#");
	}
	printf("\n");
}

int board_addStone(GameBoard* this_board, int row, int col, int state){
	int s = this_board->size;
	if (row < 0 || row >= s || col < 0 || col >= s){
		return 0;
	}
	if (this_board->draw[row * s + col] != 0){
		return 0;
	}
	this_board->draw[row * s + col] = state;
	if (checkStone(this_board, row, col, state) == 0){
		this_board->draw[row * s + col] = 0;
		return 0;
	}
	this_board->last_move = row * s + col;
	board_get_terr(this_board);
	//for test purpose
	//board_printclassify(this_board);
	return 1;
}
//remove a stone from the board
void delete_stone(GameBoard* this_board, int row, int col){
	int s = this_board->size;
	if (this_board->visited[row * s + col] == 1){
		return ;
	}
	this_board->visited[row * s + col] = 1;
	int state = this_board->draw[row * s + col];

	if (row > 0){
		if (this_board->draw[(row - 1) * s + col] == state){
			delete_stone(this_board, (row - 1), col);
			this_board->draw[(row - 1) * s + col] = 0;
		}
	} 

	if (row < s - 1){
		if (this_board->draw[(row + 1) * s + col] == state){
			delete_stone(this_board, (row + 1), col);
			this_board->draw[(row + 1) * s + col] = 0;
		}
	}

	if (col > 0){
		if (this_board->draw[row * s + col - 1] == state){
			delete_stone(this_board, row, col-1);
			this_board->draw[row * s + col - 1] = 0;
		}
	}

	if (col < s - 1){
		if (this_board->draw[row * s + col + 1] == state){
			delete_stone(this_board, row, col+1);
			this_board->draw[row * s + col + 1] = 0;
		}
	}
	this_board->draw[row * s + col] = 0;
}
//zhao qi
int get_liberties(GameBoard* this_board, int row, int col){
	int cnt = 0;
	int s = this_board->size;
	if (this_board->visited[row * s + col] == 1){
		return 0;
	}
	this_board->visited[row * s + col] = 1;
	int state = this_board->draw[row * s + col];

	if (row > 0){
		if (this_board->draw[(row - 1) * s + col] == state){
			cnt += get_liberties(this_board, (row - 1), col);
		} else {
			cnt += (this_board->draw[(row - 1) * s + col] == 0);
		}
	} 

	if (row < s - 1){
		if (this_board->draw[(row + 1) * s + col] == state){
			cnt += get_liberties(this_board, row + 1, col);
		} else {
			cnt += (this_board->draw[(row + 1) * s + col] == 0);
		}
	}

	if (col > 0){
		if (this_board->draw[row * s + col - 1] == state){
			cnt += get_liberties(this_board, row, col-1);
		} else {
			cnt += (this_board->draw[row * s + col - 1] == 0);
		}
	}

	if (col < s - 1){
		if (this_board->draw[row * s + col + 1] == state){
			cnt += get_liberties(this_board, row, col+1);
		} else {
			cnt += (this_board->draw[row * s + col + 1] == 0);
		}
	}
	return cnt;
}

int checkStone(GameBoard* this_board, int row, int col, int state){

	int neighbors[4];
	int s = this_board->size;
    // Don't check outside the board
    if (row > 0) {neighbors[0] = (row - 1) * s + col;} else {neighbors[0] = -1;}
    if (row < s - 1) {neighbors[1] = (row + 1) * s + col;} else {neighbors[1] = -1;}
    if (col > 0) {neighbors[2] = row * s + col - 1;} else {neighbors[2] = -1;}
    if (col < s - 1) {neighbors[3] = row * s + col + 1;} else {neighbors[3] = -1;}

	int flag = 1;
	if (get_liberties(this_board, row, col) == 0){
		flag = 0;
	}

	int cur_row, cur_col;
	for (int idx = 0; idx < 4; idx++){
		if (neighbors[idx] != -1 && this_board->draw[neighbors[idx]] == -state){
			cur_row = neighbors[idx] / s;
			cur_col = neighbors[idx] % s;
			clear_visited(this_board);
			if (get_liberties(this_board, cur_row, cur_col) == 0){
				clear_visited(this_board);
				delete_stone(this_board, cur_row, cur_col);
				flag = 1;
			}
		}
	}
	return flag;
}

int power_s(int x, int y){
    int p = 1;
    while (y > 0){
        p = p * x;
        y -= 1;
    }
    return p;
}

int board_monte_carlo(GameBoard* this_board, int n){
	int s = this_board->size;
    int ss = s;
    if (n == 2 and s == 19) ss = 8;
    if (n == 3 and s == 9) ss = 5; 
    if (n == 3 and s == 19) ss = 4;
    int num = power_s(ss, 2*n);
    int partial_num = int(num / (ss * ss));

    int startx = 0;
    int starty = 0;
    int last_row = this_board->last_move / s;
    int last_col = this_board->last_move % s;
    if (last_row + int(ss / 2) >= s){startx = s - ss;}
    else if (last_row - int(ss / 2) > 0) {startx = last_row - int(ss/2);}

    if (last_col + int(ss / 2) >= s){starty = s - ss;}
    else if (last_col - int(ss / 2) > 0) {starty = last_col - int(ss/2);}

	int max_pos = rand() % (s * s);
	float max_val = -101.0;

	int next_step;
	for (int ii=startx; ii<startx + ss; ii++){
		for (int jj=starty; jj<starty + ss; jj++){
			next_step = ii * s + jj;
			if (this_board->draw[next_step] == 0){
				int local_cnt = 0;
				int local_sum = 0;
				for (int p = 0; p < partial_num; p++){
					GameBoard* next_board = new GameBoard;
					board_construct(next_board, s);
					for (int i=0; i<s; i++){
						for (int j=0; j<s; j++){
							if (this_board->draw[i*s+j] != 0){
								if (this_board->draw[i*s+j] == 1){
									board_addStone(next_board, i, j, 1);
								} else {
									board_addStone(next_board, i, j, -1);
								}
							}
						}
					}

					int flag = board_addStone(next_board, next_step / s, next_step % s, -1);
					int next_next;
					int type = -1;
					if (flag == 1){
						for (int k=0; k<n-1; k++){
							next_next = rand() % (s * s);
							flag = board_addStone(next_board, next_next / s, next_next % s, type);
							type *= (-1);
							if (flag == 0) break;
						}
					}
					if (flag != 0){
						board_get_terr(next_board);
						int w_count = 0;
						for (int i=0; i<s; i++){
							for (int j=0; j<s; j++){
								if (next_board->classify[i*s+j] == 1){
									w_count -= 1;
								} else if (next_board->classify[i*s+j] == -1){
									w_count += 1;
								}
							}
						}
						local_cnt += 1;
						local_sum += w_count;
					}
				delete next_board;
				}
				if (float(local_sum) / local_cnt > max_val){
            		max_val = float(local_sum) / local_cnt;
            		max_pos = next_step;
        		}
			}
		}
	}

	return max_pos;
}

void board_printclassify(GameBoard* this_board){
	int s = this_board->size;
	for (int i = 0; i < s; i++){
		for (int j = 0; j < s; j++){
			if (this_board->classify[i*s+j] != 0){
				if (this_board->classify[i*s+j] == 1){
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

void board_get_terr(GameBoard* this_board) {
	int s = this_board->size;

	for (int i=0; i< s*s; i++){
		this_board->eval[i] = 0;
	}

	int idx, dist, diff;
	for (int r = 0; r < s; r++){
		for (int c = 0; c < s; c++){
			idx = r * s + c;
			if (this_board->draw[idx] != 0){
				diff = this_board->draw[idx];
				for(int i = max_of(r - 4, 0); i < min_of(r + 5, s); i++){
					for(int j = max_of(c - 4, 0); j < min_of(c + 5, s); j++){
						dist = abso(r - i) + abso(c - j);
						this_board->eval[i * s + j] += diff * mapping(dist);
					}
				}
			}
		}
	}

	for(int i = 0; i < s * s; i++) {
    	if(this_board->draw[i] == 1) {
      		if(this_board->eval[i] <= 0) this_board->classify[i] = -1; //dead
      		else this_board->classify[i] = 1;
    	}
    	else if(this_board->draw[i] == -1) {
      		if(this_board->eval[i] >= 0) this_board->classify[i] = 1; //dead
      		else this_board->classify[i] = -1;
    	}
    	else if(this_board->eval[i] > 0) this_board->classify[i] = 1;
    	else if(this_board->eval[i] < 0) this_board->classify[i] = -1;
    	else this_board->classify[i] = 0;
  	}
}