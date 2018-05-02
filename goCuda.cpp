#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include <iostream>
#include <string>
//#include "mygameboard.cpp"
#include "board.h"
#include <ctime>
//stones
#define WHITE (-1)
#define BLACK (1)
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

using namespace std;
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

int Monte_Carlo_Cuda(GameBoard* this_board, int n);

// no-interface version
int main(int argc, char** argv)
{
    int size = 19;
    GameBoard* board = new GameBoard;

    ofstream myfile ("time_cuda.txt");
    if (myfile.is_open())
    {
        printf("file opened\n");
    }
    
    buildBoard(board,size);
    int row, col, next_move,step;
    step = 0;

    cin >> row;
    while (row != -1){
        cin >> col;
       step += 1;
        addStone(board, row, col, 1);

       	std::clock_t start;
       	double duration = 0;
        next_move = Monte_Carlo_Cuda(board, 3);
        duration = (std::clock() - start)/(double)CLOCKS_PER_SEC;
        printf("add white stone %d\n", next_move);

        while(addStone(board, next_move/size, next_move%size, -1) ==0 ){
            next_move = rand() % (size * size);
        }
        printBoard(board);
        printf("Time is  %f\n", duration);
        printf("Step is  %d\n", step);
        myfile << duration << " "<<step<<endl; 
        cin >> row;
    }
    
    myfile.close();
    delete board;
    return 0;
}