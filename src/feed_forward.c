#include "mkl_cblas.h"
#include "mkl_vml_functions.h"
#include "mnist.h"
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

void copy_mat(Matrix *x, Matrix *y) {
  copy(x->rows * x->cols, x->data, y->data);
}

Vector *apply_layer(Matrix *M, Vector *x, Vector *biases) {
  int m = M->rows;
  int n = M->cols;

  Vector *biases_copy = empty_vector(biases->size);
  copy(biases->size, biases->data, biases_copy->data);

  cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, M->data, n, x->data, 1,
              1.0, biases_copy->data, 1);

  free_vector(x);

  return biases_copy;
}

Matrix *rep_n(Vector *v, int n) {
  Matrix *m = empty_matrix(v->size, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < v->size; j++) {
      m->data[i * v->size + j] = v->data[j];
    }
  }

  return m;
}

Matrix *apply_layer_batch(Matrix *M, Matrix *x, Vector *biases) {
  int m = M->rows;
  int n = x->cols;
  int k = M->cols;

  Matrix *biases_copy = rep_n(biases, x->cols);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, M->data, k, x->data, n, 1.0, biases_copy->data, n);

  return biases_copy;
}

double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

double sigmoid_prime(double z) { return sigmoid(z) * (1 - sigmoid(z)); }

void sigmoid_vec(Vector *v) {
  for (int i = 0; i < v->size; i++) {
    v->data[i] = sigmoid(v->data[i]);
  }
}

void sigmoid_mat(Matrix *m) {
  for (int i = 0; i < m->rows * m->cols; i++) {
    m->data[i] = sigmoid(m->data[i]);
  }
}

void sigmoid_prime_vec(Vector *v) {
  for (int i = 0; i < v->size; i++) {
    v->data[i] = sigmoid_prime(v->data[i]);
  }
}

void sigmoid_prime_mat(Matrix *m) {
  for (int i = 0; i < m->rows * m->cols; i++) {
    m->data[i] = sigmoid_prime(m->data[i]);
  }
}

Vector *apply_neural_network(NeuralNetwork *network, Vector *input) {
  for (int i = 0; i < network->num_layers - 1; i++) {
    input = apply_layer(network->weights[i], input, network->biases[i]);
    sigmoid_vec(input);
  }

  return input;
}

// Updates `output` to contain the error of the last layer
void initial_error(Matrix *output, Vector *expected) {
  for (int i = 0; i < output->cols; i++) {
    int index = i * output->cols + expected->data[i];
    output->data[index] = output->data[index] - 1;
  }
}

void shift_values(Matrix *weight, Vector *bias, Matrix *error,
                  Matrix *activation, double learning_rate) {
  for (int i = 0; i < bias->size; i++) {
    double average_gradient = 0;
    for (int j = 0; j < error->cols; j++) {
      average_gradient += error->data[i * error->cols + j];
    }
    average_gradient /= error->cols;

    bias->data[i] -= learning_rate * average_gradient;
  }

  for (int i = 0; i < weight->rows; i++) {
    for (int j = 0; j < weight->cols; j++) {
      double average_gradient = 0;
      
      for (int k = 0; k < error->cols; k++) {
        average_gradient += error->data[i * error->cols + k] *
                            activation->data[j * activation->cols + k];
      }
      average_gradient /= error->cols;

      int index = i * weight->cols + j;
      weight->data[index] -= learning_rate * average_gradient;
    }
  }
}

void calculate_error(Matrix *error, Matrix *input, Matrix *weight) {
  int m = weight->rows;
  int n = weight->cols;
  int k = error->cols;
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0, weight->data, m, error->data, n, 0.0, input->data, n);

  error->rows = input->rows;

  sigmoid_prime_mat(input);

  vdMul(input->cols * input->rows, input->data, error->data, error->data);
}

void backpropagate(NeuralNetwork *network, Matrix **inputs,
                   Matrix **activations, Matrix *output, Vector *expected,
                   double learning_rate) {
  Matrix *error = empty_matrix(network->max_size, output->cols);
  error->rows = output->rows;
  copy_mat(output, error);
  initial_error(error, expected);

  for (int i = network->num_layers - 2; i >= 0; i--) {
    Matrix *weight = network->weights[i];
    Vector *bias = network->biases[i];

    shift_values(weight, bias, error, activations[i], learning_rate);

    if (i > 0) {
      calculate_error(error, inputs[i - 1], weight);
    }
  }

  free_matrix(error);
}

void train_batch(NeuralNetwork *network, Matrix *batch, Vector *expected,
                 double learning_rate) {
  Matrix **inputs = malloc((network->num_layers - 1) * sizeof(Matrix *));
  Matrix **activations = malloc(network->num_layers * sizeof(Matrix *));
  activations[0] = batch;
  for (int i = 0; i < network->num_layers - 1; i++) {
    batch = apply_layer_batch(network->weights[i], batch, network->biases[i]);

    Matrix *layer_input = empty_matrix(batch->rows, batch->cols);
    copy_mat(batch, layer_input);
    inputs[i] = layer_input;

    sigmoid_mat(batch);
    activations[i + 1] = batch;
  }

  backpropagate(network, inputs, activations, batch, expected, learning_rate);

  for (int i = 0; i < network->num_layers - 1; i++) {
    free_matrix(inputs[i]);
  }

  for (int i = 0; i < network->num_layers; i++) {
    free_matrix(activations[i]);
  }

  free(inputs);
  free(activations);
}

Matrix *create_batch(int size, double (*images)[784]) {
  Matrix *batch = empty_matrix(784, size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < 784; j++) {
      batch->data[i * 784 + j] = images[i][j];
    }
  }

  return batch;
}

Vector *create_expected(int size, int *labels) {
  Vector *expected = empty_vector(size);
  for (int i = 0; i < size; i++) {
    expected->data[i] = labels[i];
  }

  return expected;
}

int main(void) {
  load_mnist();

  double learning_rate = 0.01;
  int epochs = 1;
  int batch_size = 10;

  int sizes[] = {784, 30, 10};
  NeuralNetwork *network = empty_neural_network(3, sizes);
  initialise_neural_network(network);

  for (int i = 0; i < epochs; i++) {
    for (int j = 0; j < 1; j++) {
      Matrix *batch = create_batch(batch_size, train_image + i * batch_size);
      Vector *expected = create_expected(batch_size, train_label + i * batch_size);
      train_batch(network, batch, expected, learning_rate);
      free_vector(expected);
    }
  }

  print_neural_network(network);

  Vector *input = new_vector(784, test_image[1]);
  Vector *output = apply_neural_network(network, input);

  print_vector(output);
  printf("%d\n", test_label[1]);

  free_vector(output);
  free_neural_network(network);

  return EXIT_SUCCESS;
}
