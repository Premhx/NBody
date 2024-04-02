#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N 5000
#define STEPS 3000
#define DT 0.0001
#define THETA 0.5 // Barnes-Hut opening angle

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

typedef struct QuadtreeNode {
    double center[2];
    double size;
    Body *body;
    struct QuadtreeNode *children[4];
} QuadtreeNode;

// Function prototypes
QuadtreeNode *create_quadtree(double center[2], double size);
void insert_body(QuadtreeNode *node, Body *body);
void update_tree(QuadtreeNode *node);
void calculate_force(Body *body, QuadtreeNode *node, double theta);
void update_bodies(Body *bodies, int num_bodies, double dt, QuadtreeNode *quadtree);
void simulate_and_save(int seed, int steps, const char *filename);

// Your existing functions remain unchanged

// Main function
int main() {
    simulate_and_save(3, STEPS, "simulation_coords.csv");
    return 0;
}

// Function to create a new quadtree node
QuadtreeNode *create_quadtree(double center[2], double size) {
    QuadtreeNode *node = (QuadtreeNode *)malloc(sizeof(QuadtreeNode));
    if (node == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    node->center[0] = center[0];
    node->center[1] = center[1];
    node->size = size;
    node->body = NULL;
    for (int i = 0; i < 4; i++) {
        node->children[i] = NULL;
    }
    return node;
}

// Function to insert a body into the quadtree
void insert_body(QuadtreeNode *node, Body *body) {
    if (node->body == NULL) {
        node->body = body;
        return;
    }

    if (node->children[0] == NULL) {
        // Subdivide if not already subdivided
        double newSize = node->size / 2.0;
        double x = node->center[0];
        double y = node->center[1];
        node->children[0] = create_quadtree((double[]){x - newSize / 2, y - newSize / 2}, newSize);
        node->children[1] = create_quadtree((double[]){x + newSize / 2, y - newSize / 2}, newSize);
        node->children[2] = create_quadtree((double[]){x - newSize / 2, y + newSize / 2}, newSize);
        node->children[3] = create_quadtree((double[]){x + newSize / 2, y + newSize / 2}, newSize);

        // Move the existing body down to appropriate child node
        Body *existingBody = node->body;
        node->body = NULL;
        insert_body(node, existingBody);
    }

    // Insert new body into appropriate child node
    double posX = body->pos[0];
    double posY = body->pos[1];
    if (posX < node->center[0]) {
        if (posY < node->center[1]) {
            insert_body(node->children[0], body);
        } else {
            insert_body(node->children[2], body);
        }
    } else {
        if (posY < node->center[1]) {
            insert_body(node->children[1], body);
        } else {
            insert_body(node->children[3], body);
        }
    }
}

// Function to update the quadtree with new body positions
void update_tree(QuadtreeNode *node) {
    if (node->children[0] != NULL) {
        // Update children first
        for (int i = 0; i < 4; i++) {
            update_tree(node->children[i]);
        }

        // Calculate center of mass and total mass
        double totalMass = 0.0;
        double centerOfMass[2] = {0.0, 0.0};
        for (int i = 0; i < 4; i++) {
            if (node->children[i]->body != NULL) {
                Body *childBody = node->children[i]->body;
                double childMass = childBody->mass;
                totalMass += childMass;
                centerOfMass[0] += childBody->pos[0] * childMass;
                centerOfMass[1] += childBody->pos[1] * childMass;
            }
        }
        if (totalMass > 0.0) {
            centerOfMass[0] /= totalMass;
            centerOfMass[1] /= totalMass;
        }

        // Update current node with center of mass and total mass
        node->body = (Body *)malloc(sizeof(Body));
        node->body->pos[0] = centerOfMass[0];
        node->body->pos[1] = centerOfMass[1];
        node->body->mass = totalMass;
    }
}

// Function to calculate force acting on a body due to distant bodies using Barnes-Hut algorithm
void calculate_force(Body *body, QuadtreeNode *node, double theta) {
    double dX = node->body->pos[0] - body->pos[0];
    double dY = node->body->pos[1] - body->pos[1];
    double distance = sqrt(dX * dX + dY * dY);
    if (node->size / distance < theta) {
        // Use node's center of mass as approximation
        double force = (body->mass * node->body->mass) / (distance * distance);
        body->vel[0] += force * (dX / distance) * DT;
        body->vel[1] += force * (dY / distance) * DT;
    } else {
        // Recursively apply the Barnes-Hut algorithm
        for (int i = 0; i < 4; i++) {
            if (node->children[i] != NULL) {
                calculate_force(body, node->children[i], theta);
            }
        }
    }
}

// Function to update bodies using the Barnes-Hut algorithm
void update_bodies(Body *bodies, int num_bodies, double dt, QuadtreeNode *quadtree) {
    // Update the quadtree with new body positions
    for (int i = 0; i < num_bodies; i++) {
        insert_body(quadtree, &bodies[i]);
    }
    update_tree(quadtree);

    // Calculate forces and update velocities/positions of bodies
    for (int i = 0; i < num_bodies; i++) {
        calculate_force(&bodies[i], quadtree, THETA);
        bodies[i].pos[0] += bodies[i].vel[0] * dt;
        bodies[i].pos[1] += bodies[i].vel[1] * dt;
    }
}

// Function to simulate and save results
void simulate_and_save(int seed, int steps, const char *filename) {
    srand(seed);
    Body bodies[N];
    int i, j;
    int progress_interval = steps / 100;  // Adjust interval for desired progress updates
    double start_time = omp_get_wtime();

    // Initialize quadtree
    double center[2] = {0.0, 0.0}; // Assuming center at origin
    double size = 100.0; // Adjust size according to your simulation space
    QuadtreeNode *quadtree = create_quadtree(center, size);

    for (i = 0; i < N; i++) {
        rand_body(&bodies[i]);
    }

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    for (i = 0; i < steps; i++) {
        for (j = 0; j < N; j++) {
            fprintf(file, "%lf,%lf\n", bodies[j].pos[0], bodies[j].pos[1]);
        }
        fprintf(file, "\n");
        update_bodies(bodies, N, DT, quadtree);

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

// Your rand_body function remains unchanged

// Your rand_double function remains unchanged
