#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <iostream>
#include <string>
#include "GameBoard.cpp"
using namespace std;

int main(int argc, char** argv)
{
    GameBoard* board = new GameBoard;
    board_construct(board, 9);
    grid_printboard(board->game_grid);

    int row, col, cur_state;
    cin >> row;
    cur_state = 1;
    while (row != -1){
        cin >> col;
        board_addStone(board, row, col, cur_state);
        int move = board_monte_carlo(board);
        if (move != -1){
            board_addStone(board, move / 9 + 1, move % 9 + 1, 0);
        } else {
            printf("Ren Shu\n");
            break;
        }
        // if (cur_state == 1){
        //     cur_state = 0;
        // } else {
        //     cur_state = 1;
        // }
        cin >> row;
    }
    return 0;
}
