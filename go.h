#ifndef GO_H
#define GO_H
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include <iostream>
#include <string>
#include <ctime>
#include <stdlib.h>
//stones
#define WHITE (-1)
#define BLACK (1)
//operators;
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define abs(a) ((a)>=(0)?(a):(-a))

//do not use cpp stl since cuda won't support it
struct GameBoard{
	int size;
	int current_player_state; // 1 for black, -1 for white
	int last_move;
	int draw[361];
	int eval[361];
	int classify[361];
	int visited[361];
};

//evaluate opponents threa
inline int map(int dist);
//count score
void getTerr(GameBoard* this_board);
//check if the stone is dead
int checkStone(GameBoard* myboard, int row, int col, int state);
//clean the board for dfs
inline void cleanBoard(GameBoard* myboard);
//construct a board
void buildBoard(GameBoard* myboard, int size);
//ui, printout the board
void printBoard(GameBoard* myboard);
//add one stone to the board
int addStone(GameBoard* myboard, int row, int col, int state);
//remove a stone from the board
void deleteStone(GameBoard* myboard, int row, int col);
// a part, check the stone is dead or not
int countLiberty(GameBoard* myboard, int row, int col);
//check the stone is dead or not
int checkStone(GameBoard* myboard, int row, int col, int state);
//compute power of a number
inline int raisePwr(int num, int times);
//using cuda to add a stone, parallizing the check score part
int cudaaddStone(GameBoard* myboard, int row, int col, int state);

//kernel function for serial MonteCarlo
int serialkernelMonteCarlo(GameBoard* myboard, int n);
//cuda
int cudaMonteCarlo(GameBoard* this_board, int n);
//removed the count territory function from that
//int cudaaddStone(GameBoard* myboard, int row, int col, int state);

#endif