#ifndef WORLD_H
#define WORLD_H

typedef struct {
    double population;
    double energy;
    double pollution;
    double renewable_ratio;
    double economy;
} WorldState;

void initializeWorld(WorldState *state);
void updateWorld(WorldState *state);

#endif
