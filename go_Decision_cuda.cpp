#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include <iostream>
#include <string>
#include "mygameboard.cpp"
#include "readfile.h"
#include <ctime>
using namespace std;

int Monte_Carlo_Cuda(GameBoard* this_board, int n);

// no-interface version
int main(int argc, char** argv)
{
    int size = 19;
    GameBoard* board = new GameBoard;
    //board_construct(board, size);
    buildBoard(board,size);
    int row, col, next_move;
    cin >> row;
    while (row != -1){
        cin >> col;
        //board_addStone(board, row, col, 1);
        addStone(board, row, col, 1);
        //next_move = board_monte_carlo(board, 2);
       	std::clock_t start;
       	double duration = 0;
        next_move = Monte_Carlo_Cuda(board, 3);
        duration = (std::clock() - start)/(double)CLOCKS_PER_SEC;
        printf("add white stone %d\n", next_move);
        //while (board_addStone(board, next_move / size, next_move % size, -1) == 0){
        while(addStone(board, next_move/size, next_move%size, -1) ==0 ){
            next_move = rand() % (size * size);
        }
        printBoard(board);
        printf("Time is  %f\n", duration);
        cin >> row;
    }
    delete board;
    return 0;
}
