#ifndef READFILE_H
#define READFILE_H

struct GameBoard{
	int size;
	int current_player_state; // 1 for black, -1 for white
	int last_move;
	int draw[361];
	int eval[361];
	int classify[361];
	int visited[361];
};

#endif