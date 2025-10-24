#include "energy.h"

void updateEnergy(WorldState *state) {
    // Energy consumption increases with population and economy
    double base_energy = 1000.0;
    state->energy = base_energy * (state->population / 8000000000) * (state->economy / 100);
    // Adjust based on renewable ratio (renewables are more efficient)
    state->energy *= (1 + 0.2 * state->renewable_ratio);
}
