#include "agent.h"
#include <stdlib.h>

void updateAgents(WorldState *state) {
    // Simple agent: Governments invest in renewables if pollution is high
    if (state->pollution > 60) {
        double investment = 0.1 * state->economy;
        state->renewable_ratio += investment / 1000;  // Increase ratio
        state->economy -= investment;  // Cost
        if (state->renewable_ratio > 1) state->renewable_ratio = 1;
    }
}
