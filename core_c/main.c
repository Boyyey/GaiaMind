#include <stdio.h>
#include <stdlib.h>
#include "world.h"
#include "agent.h"
#include "energy.h"

#define SIMULATION_YEARS 100
#define TICKS_PER_YEAR 12  // Monthly ticks

int main() {
    WorldState state;
    initializeWorld(&state);
    FILE *logFile = fopen("../results/world_log.csv", "w");
    if (!logFile) {
        perror("Error opening log file");
        return 1;
    }
    fprintf(logFile, "Year,Population,Energy,Pollution,RenewableRatio,Economy\n");

    for (int year = 0; year < SIMULATION_YEARS; year++) {
        for (int tick = 0; tick < TICKS_PER_YEAR; tick++) {
            updateWorld(&state);
            updateAgents(&state);
            updateEnergy(&state);
        }
        // Log annual data
        fprintf(logFile, "%d,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                year, state.population, state.energy, state.pollution,
                state.renewable_ratio, state.economy);
        printf("Year %d: Pop=%.0f, Energy=%.2f, Pollution=%.2f, Economy=%.2f\n",
               year, state.population, state.energy, state.pollution, state.economy);
    }

    fclose(logFile);
    printf("Simulation complete. Check results/world_log.csv\n");
    return 0;
}
