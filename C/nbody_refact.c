#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define DEFAULT_N 5000
#define DEFAULT_STEPS 2000
#define DEFAULT_DT 0.0001
#define DEFAULT_FILENAME "simulation_coords.csv"
#define DEFAULT_SEED 3

typedef struct {
    double x;
    double y;
} Vector;

typedef struct {
    Vector pos;
    Vector vel;
    double mass;
} Body;

double rand_double() {
    return (double)rand() / RAND_MAX;
}

Vector create_vector(double x, double y) {
    Vector v = {x, y};
    return v;
}

Vector add_vectors(Vector v1, Vector v2) {
    Vector result = {v1.x + v2.x, v1.y + v2.y};
    return result;
}

Vector multiply_scalar(Vector v, double scalar) {
    Vector result = {v.x * scalar, v.y * scalar};
    return result;
}

double calculate_distance(Vector v1, Vector v2) {
    double dx = v2.x - v1.x;
    double dy = v2.y - v1.y;
    return sqrt(dx * dx + dy * dy);
}

Vector calculate_force(Vector pos1, Vector pos2, double m1, double m2) {
    const double G = 1;
    // const double G = 6.67430e-11;

    Vector r = {pos2.x - pos1.x, pos2.y - pos1.y};
    double distance = fmax(calculate_distance(pos1, pos2), 0.0001);
    double magnitude = G * m1 * m2 / (distance * distance);
    return multiply_scalar(r, magnitude / distance);
}

void update_body_velocity(Body *body, Vector force, double dt) {
    body->vel = add_vectors(body->vel, multiply_scalar(force, dt / body->mass));
}

void update_body_position(Body *body, double dt) {
    body->pos = add_vectors(body->pos, multiply_scalar(body->vel, dt));
}

void update_bodies(Body *bodies, int num_bodies, double dt) {
    #pragma omp parallel for
    for (int i = 0; i < num_bodies; i++) {
        Vector total_force = {0, 0};
        for (int j = 0; j < num_bodies; j++) {
            if (i != j) {
                Vector force = calculate_force(bodies[i].pos, bodies[j].pos, bodies[i].mass, bodies[j].mass);
                total_force = add_vectors(total_force, force);
            }
        }
        update_body_velocity(&bodies[i], total_force, dt);
    }

    #pragma omp parallel for
    for (int i = 0; i < num_bodies; i++) {
        update_body_position(&bodies[i], dt);
    }
}

void simulate_and_save(int N, int steps,  double dt, const char *filename,int seed) {
    printf("Simulation conditions:\n");
    printf("Number of particles: %d\n", N);
    printf("Number of steps: %d\n", steps);
    printf("Time step (dt): %lf\n", dt);

    srand(seed);
    printf("Memory allocation.\n");
    Body *bodies = malloc(N * sizeof(Body));
    if (bodies == NULL) {
        printf("Memory allocation failed.\n");
        return;
    }

    printf("Generating bodies data.\n");
        // Define the range for position coordinates
    double lower_bound = -10.0; // Example lower bound
    double upper_bound = 10.0;  // Example upper bound
    for (int i = 0; i < N; i++) {
        double rand_x = rand_double() * (upper_bound - lower_bound) + lower_bound;
        double rand_y = rand_double() * (upper_bound - lower_bound) + lower_bound;
        bodies[i].pos = create_vector(rand_x, rand_y);

        // bodies[i].pos = create_vector(rand_double(), rand_double());
        // bodies[i].vel = create_vector(rand_double(), rand_double());
        bodies[i].vel = create_vector(0.0, 0.0);
        
        bodies[i].mass = rand_double();
    }

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        free(bodies);
        return;
    }
    printf("Starting simulation.\n");
    double start_time = omp_get_wtime(); // Start timing

    int progress_interval = steps / 100; // For progress indication

    for (int i = 0; i < steps; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(file, "%lf,%lf\n", bodies[j].pos.x, bodies[j].pos.y);
        }
        fprintf(file, "\n");
        update_bodies(bodies, N, dt);

        // Progress indication
        if ((i + 1) % progress_interval == 0 || i == steps - 1) {
            double elapsed_time = omp_get_wtime() - start_time;
            int progress = (i + 1) * 100 / steps;
            int eta_seconds = (int)((100.0 - progress) / progress * elapsed_time);
            int eta_hours = eta_seconds / 3600;
            int eta_minutes = (eta_seconds % 3600) / 60;
            int eta_secs = eta_seconds % 60;

            printf("\rProgress: %d%%  ETA: %02d:%02d:%02d", progress, eta_hours, eta_minutes, eta_secs);
            fflush(stdout); // Flush output to ensure it's visible immediately
        }
    }

    double final_time = omp_get_wtime() - start_time;
    printf("\nSimulation finished.\n");
    printf("Total time for simulation: %lf seconds\n", final_time);

    fclose(file);
    free(bodies);
}




int main(int argc, char *argv[]) {
    int num_bodies = DEFAULT_N;
    int steps = DEFAULT_STEPS;
    double dt = DEFAULT_DT;
    const char *filename = DEFAULT_FILENAME;
    int seed = DEFAULT_SEED;

    // Parse command line arguments if provided
    if (argc > 1) {
        num_bodies = atoi(argv[1]);
    }
    if (argc > 2) {
        steps = atoi(argv[2]);
    }
    if (argc > 3) {
        dt = atof(argv[3]);
    }
    if (argc > 4) {
        filename = argv[4];
    }
    if (argc > 5) {
        seed = atoi(argv[5]);
    }

    // Call the simulation function
    simulate_and_save(num_bodies, steps, dt, filename, seed);

    return 0;
}
