#include "mkl.h"
#include "mkl_cblas.h"
#include "mkl_vml_functions.h"
#include "mnist.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

struct Vector {
  int size;
  double *data;
};

typedef struct Vector Vector;

struct Matrix {
  int rows;
  int cols;
  double *data;
};

typedef struct Matrix Matrix;

struct NeuralNetwork {
  int num_layers;
  int *sizes;
  Vector **biases;
  Matrix **weights;
  int max_size;
};

typedef struct NeuralNetwork NeuralNetwork;

Vector *empty_vector(int size) {
  Vector *v = malloc(sizeof(Vector));
  v->size = size;
  v->data = malloc(size * sizeof(double));
  return v;
}

Vector *new_vector(int size, double *data) {
  Vector *v = empty_vector(size);
  for (int i = 0; i < size; i++) {
    v->data[i] = data[i];
  }
  return v;
}

void free_vector(Vector *v) {
  free(v->data);
  free(v);
}

void print_vector(Vector *v) {
  printf("[");
  for (int i = 0; i < v->size; i++) {
    printf("%f", v->data[i]);
    if (i < v->size - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

Matrix *empty_matrix(int rows, int cols) {
  Matrix *m = malloc(sizeof(Matrix));
  m->rows = rows;
  m->cols = cols;
  m->data = malloc(rows * cols * sizeof(double));
  return m;
}

Matrix *new_matrix(int rows, int cols, double *data) {
  Matrix *m = empty_matrix(rows, cols);
  for (int i = 0; i < rows * cols; i++) {
    m->data[i] = data[i];
  }
  return m;
}

void free_matrix(Matrix *m) {
  free(m->data);
  free(m);
}

NeuralNetwork *empty_neural_network(int num_layers, int *sizes) {
  NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
  nn->num_layers = num_layers;
  nn->sizes = sizes;
  nn->biases = malloc((num_layers - 1) * sizeof(Vector));
  nn->weights = malloc((num_layers - 1) * sizeof(Matrix));

  int max = 0;
  for (int i = 0; i < num_layers - 1; i++) {
    if (sizes[i + 1] > max) {
      max = sizes[i + 1];
    }

    nn->biases[i] = empty_vector(sizes[i + 1]);
    nn->weights[i] = empty_matrix(sizes[i + 1], sizes[i]);
  }

  nn->max_size = max;

  return nn;
}

double random_double(void) { return (rand() / (double)RAND_MAX) * 2 - 1; }

void initialise_neural_network(NeuralNetwork *network) {
  for (int i = 0; i < network->num_layers - 1; i++) {
    Vector *bias = network->biases[i];
    Matrix *weight = network->weights[i];
    for (int j = 0; j < bias->size; j++) {
      bias->data[j] = random_double();
    }

    for (int j = 0; j < weight->cols * weight->rows; j++) {
      weight->data[j] = random_double();
    }
  }
}

void print_neural_network(NeuralNetwork *network) {
  for (int i = 0; i < network->num_layers - 1; i++) {
    printf("Layer %d\n", i);
    printf("Biases: ");
    print_vector(network->biases[i]);
  }
}

void free_neural_network(NeuralNetwork *network) {
  for (int i = 0; i < network->num_layers - 1; i++) {
    free_vector(network->biases[i]);
    free_matrix(network->weights[i]);
  }

  free(network->biases);
  free(network->weights);
  free(network);
}

void copy(int n, double *x, double *y) {
  for (int i = 0; i < n; i++) {
    y[i] = x[i];
  }
}

void copy_vec(Vector *x, Vector *y) { copy(x->size, x->data, y->data); }

Vector *apply_layer(Matrix *M, Vector *x, Vector *biases, bool free_x) {
  int m = M->rows;
  int n = M->cols;

  Vector *biases_copy = empty_vector(biases->size);
  copy(biases->size, biases->data, biases_copy->data);

  cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, M->data, n, x->data, 1,
              1.0, biases_copy->data, 1);

  if (free_x) {
    free_vector(x);
  }

  return biases_copy;
}

double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

double sigmoid_prime(double z) { return sigmoid(z) * (1 - sigmoid(z)); }

void sigmoid_vec(Vector *v) {
  for (int i = 0; i < v->size; i++) {
    v->data[i] = sigmoid(v->data[i]);
  }
}

void sigmoid_prime_vec(Vector *v) {
  for (int i = 0; i < v->size; i++) {
    v->data[i] = sigmoid_prime(v->data[i]);
  }
}

Vector *apply_neural_network(NeuralNetwork *network, Vector *input) {
  for (int i = 0; i < network->num_layers - 1; i++) {
    input = apply_layer(network->weights[i], input, network->biases[i], true);
    sigmoid_vec(input);
  }

  return input;
}

// Updates `output` to contain the error of the last layer
void initial_error(Vector *output, int expected) {
  output->data[expected] = output->data[expected] - 1;
}

void shift_values(Matrix *weight, Vector *bias, Vector *error,
                  Vector *activation, double learning_rate) {
  assert(bias->size == error->size);
  for (int i = 0; i < bias->size; i++) {
    bias->data[i] -= learning_rate * error->data[i];
  }

  assert(error->size == weight->rows);
  assert(activation->size == weight->cols);
  for (int i = 0; i < weight->rows; i++) {
    for (int j = 0; j < weight->cols; j++) {
      int index = i * weight->cols + j;
      weight->data[index] -=
          learning_rate * error->data[i] * activation->data[j];
    }
  }
}

void calculate_error(Vector *error, Vector *input, Matrix *weight) {
  cblas_dgemv(CblasRowMajor, CblasTrans, weight->cols, weight->rows, 1.0,
              weight->data, weight->cols, error->data, 1, 0.0, error->data, 1);

  error->size = input->size;

  sigmoid_prime_vec(input);

  vdMul(input->size, input->data, error->data, error->data);
}

void backpropagate(NeuralNetwork *network, Vector **inputs,
                   Vector **activations, Vector *output, int expected,
                   double learning_rate) {
  Vector *error = empty_vector(network->max_size);
  error->size = network->sizes[network->num_layers - 1];
  copy_vec(output, error);
  initial_error(error, expected);

  for (int i = network->num_layers - 2; i >= 0; i--) {
    Matrix *weight = network->weights[i];
    Vector *bias = network->biases[i];

    shift_values(weight, bias, error, activations[i], learning_rate);

    if (i > 0) {
      calculate_error(error, inputs[i - 1], weight);
    }
  }

  free_vector(error);
}

void train_case(NeuralNetwork *network, Vector *input, int expected,
                double learning_rate) {
  Vector **inputs = malloc((network->num_layers - 1) * sizeof(Vector *));
  Vector **activations = malloc(network->num_layers * sizeof(Vector *));
  activations[0] = input;
  for (int i = 0; i < network->num_layers - 1; i++) {
    input = apply_layer(network->weights[i], input, network->biases[i], false);

    Vector *layer_input = empty_vector(input->size);
    copy_vec(input, layer_input);
    inputs[i] = layer_input;

    sigmoid_vec(input);
    activations[i + 1] = input;
  }

  backpropagate(network, inputs, activations, input, expected, learning_rate);

  for (int i = 0; i < network->num_layers - 1; i++) {
    free_vector(inputs[i]);
  }

  for (int i = 0; i < network->num_layers; i++) {
    free_vector(activations[i]);
  }

  free(inputs);
  free(activations);
}

int main(void) {
  load_mnist();

  int sizes[] = {784, 30, 10};
  NeuralNetwork *network = empty_neural_network(3, sizes);
  initialise_neural_network(network);

  for (int i = 0; i < 60000; i++) {
    Vector *input = new_vector(784, train_image[i]);
    train_case(network, input, train_label[i], 0.01);
  }

  print_neural_network(network);

  Vector *input = new_vector(784, test_image[2]);
  Vector *output = apply_neural_network(network, input);

  print_vector(output);
  printf("%d\n", test_label[2]);

  free_vector(output);
  free_neural_network(network);

  return EXIT_SUCCESS;
}
