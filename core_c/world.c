#include "world.h"
#include <math.h>

void initializeWorld(WorldState *state) {
    state->population = 8000000000.0;  // 8 billion
    state->energy = 1000.0;
    state->pollution = 50.0;
    state->renewable_ratio = 0.2;  // 20% renewable
    state->economy = 100.0;
}

void updateWorld(WorldState *state) {
    // Simple population growth
    double growth_rate = 0.01 * (1 - state->pollution / 100);  // Pollution affects growth
    state->population *= (1 + growth_rate);

    // Economy based on energy and population
    state->economy = 100 * (state->energy / 1000) * (state->population / 8000000000);

    // Pollution from energy use
    double fossil_energy = state->energy * (1 - state->renewable_ratio);
    state->pollution += 0.1 * fossil_energy - 0.05 * state->renewable_ratio * state->energy;
    if (state->pollution < 0) state->pollution = 0;
    if (state->pollution > 100) state->pollution = 100;
}
