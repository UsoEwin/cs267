#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include <iostream>
#include <string>
#include "go.h"
#include <omp.h>
#include <chrono>
using namespace std;
using namespace std::chrono;


int main(int argc, char** argv)
{
    int size = 19;
   	/*
   	ofstream myfile ("time_serial.txt");
   	if (myfile.is_open())
   	{
   		printf("file opened\n");
   	}
   	*/
    GameBoard* board = new GameBoard;
    buildBoard(board,size);
    int row, col, next_move,step;
    step = 0;
    cin >> row;
    while (row != -1){
        cin >> col;
        step+=1;
        addStone(board, row, col, 1);
        
        
       	double duration = 0;
        next_move = serialkernelMonteCarlo(board, 3);

        printf("add white stone %d\n", next_move);

        
        while(addStone(board, next_move/size, next_move%size, -1) ==0 ){
            next_move = rand() % (size * size);
        }
        printBoard(board);
        printf("Time is  %f\n", duration);
        printf("Step is  %d\n", step);
        //myfile << duration << " "<<step<<endl; 
        cin >> row;
    }
    //myfile.close();
    delete board;
    return 0;
}
