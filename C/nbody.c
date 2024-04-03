#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N 10000
#define STEPS 10000
#define DT 0.0001

typedef struct {
    double pos[2];
    double vel[2];
    double mass;
} Body;

double rand_double() {
    return (double)rand() / RAND_MAX;
}

void rand_disc(double *pairs) {
    double theta = rand_double() * 2 * M_PI;
    pairs[0] = cos(theta) * sqrt(rand_double());
    pairs[1] = sin(theta) * sqrt(rand_double());
}

void rand_body(Body *body) {
    rand_disc(body->pos);
    rand_disc(body->vel);
    body->mass = rand_double();
}

void update_bodies(Body *bodies, int num_bodies, double dt) {
    int i, j;
    double acc[num_bodies][2];
    double d_min = 0.0001;

    #pragma omp parallel for private(j) shared(bodies, acc)
    for (i = 0; i < num_bodies; i++) {
        acc[i][0] = 0.0;
        acc[i][1] = 0.0;
    }

    #pragma omp parallel for private(j)
    for (i = 0; i < num_bodies; i++) {
        double p1[2] = {bodies[i].pos[0], bodies[i].pos[1]};
        double m1 = bodies[i].mass;
        for (j = i + 1; j < num_bodies; j++) {
            double p2[2] = {bodies[j].pos[0], bodies[j].pos[1]};
            double m2 = bodies[j].mass;

            double r[2] = {p2[0] - p1[0], p2[1] - p1[1]};
            double mag_sq = fmax(r[0] * r[0] + r[1] * r[1], d_min);
            double mag = sqrt(mag_sq);
            double tmp[2] = {r[0] / (mag_sq * mag), r[1] / (mag_sq * mag)};

            #pragma omp atomic
            acc[i][0] += m2 * tmp[0];
            #pragma omp atomic
            acc[i][1] += m2 * tmp[1];
            #pragma omp atomic
            acc[j][0] -= m1 * tmp[0];
            #pragma omp atomic
            acc[j][1] -= m1 * tmp[1];
        }
    }

    #pragma omp parallel for
    for (i = 0; i < num_bodies; i++) {
        bodies[i].vel[0] += acc[i][0] * dt;
        bodies[i].vel[1] += acc[i][1] * dt;
        bodies[i].pos[0] += bodies[i].vel[0] * dt;
        bodies[i].pos[1] += bodies[i].vel[1] * dt;
    }
}

void simulate_and_save(int seed, int steps, const char *filename) {
    srand(seed);
    Body bodies[N];
    int i, j;
    int progress_interval = steps / 100;  // Adjust interval for desired progress updates
    double start_time = omp_get_wtime();

    for (i = 0; i < N; i++) {
        rand_body(&bodies[i]);
    }

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }
    // fprintf(file, "step,body_id,x,y\n");
    for (i = 0; i < steps; i++) {
        for (j = 0; j < N; j++) {
            // fprintf(file, "%d,%d,%lf,%lf\n", i, j, bodies[j].pos[0], bodies[j].pos[1]);
            fprintf(file, "%lf,%lf\n", bodies[j].pos[0], bodies[j].pos[1]);

        }
        fprintf(file, "\n");
        update_bodies(bodies, N, DT);

        // Progress indication
        if ((i + 1) % progress_interval == 0 || i == steps - 1) {
            double elapsed_time = omp_get_wtime() - start_time;
            int progress = (i + 1) * 100 / steps;
            int eta_seconds = (int)((100.0 - progress) / progress * elapsed_time);
            int eta_hours = eta_seconds / 3600;
            int eta_minutes = (eta_seconds % 3600) / 60;
            int eta_secs = eta_seconds % 60;

            printf("Progress: %d%%  ETA: %02d:%02d:%02d\n", progress, eta_hours, eta_minutes, eta_secs);
        }
    }

    fclose(file);
}

int main() {
    simulate_and_save(3, STEPS, "simulation_coords.csv");
    return 0;
}
