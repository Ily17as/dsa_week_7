#include <iostream>
#include <vector>
#include <iomanip>
#include <math.h>

using namespace std;

class Matrix {
private:
    vector<vector<double>> data;
    int rows, cols;

public:
    // Constructor
    Matrix(int rows, int cols) : rows(rows), cols(cols)
    {
        data.resize(rows, vector<double>(cols, 0));
    }

    // Input operator overload
    friend istream& operator>>(istream& is, Matrix& matrix)
    {
        for (int i = 0; i < matrix.rows; ++i) {
            for (int j = 0; j < matrix.cols; ++j) {
                is >> matrix.data[i][j];
            }
        }
        return is;
    }

    // Output operator overload
    friend ostream& operator<<(ostream& os, const Matrix& matrix)
    {
        for (const auto& row : matrix.data) {
            for (const auto& elem : row) {
                os << elem << ' ';
            }
            os << "\n";
        }
        return os;
    }

    // Assignment operator overload
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            data = other.data;
        }
        return *this;
    }

    // Addition operator overload
    Matrix operator+(const Matrix& other)
    {
        if (rows != other.rows || cols != other.cols) {
            cout << "Error: the dimensional problem occurred" << endl;
            return {0, 0};
        } else {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    result.data[i][j] = data[i][j] + other.data[i][j];
                }
            }
            return result;
        }
    }

    // Subtraction operator overload
    Matrix operator-(const Matrix& other)
    {
        if (rows != other.rows || cols != other.cols) {
            cout << "Error: the dimensional problem occurred" << endl;
            return {0, 0};
        } else {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    result.data[i][j] = data[i][j] - other.data[i][j];
                }
            }
            return result;
        }
    }

    // Multiplication operator overload
    Matrix operator*(const Matrix& other)
    {
        if (cols != other.rows) {
            cout << "Error: the dimensional problem occurred" << endl;
            return {0, 0};
        } else {
            Matrix result(rows, other.cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < other.cols; ++j) {
                    for (int k = 0; k < cols; ++k) {
                        result.data[i][j] += data[i][k] * other.data[k][j];
                    }
                }
            }
            return result;
        }
    }

    // Transpose matrix
    Matrix transpose()
    {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }
};

class SquareMatrix{
public:
    vector<vector<double>> data;
    int size;

public:
    // Constructor
    SquareMatrix(int size) : size(size) {
        data.resize(size, vector<double>(size, 0));
    }

    // Output operator overload
    friend ostream& operator<<(ostream& os, const SquareMatrix& matrix) {
        for (const auto& row : matrix.data) {
            for (int val : row) {
                os << val << ' ';
            }
            os << '\n';
        }
        return os;
    }

    friend istream& operator>>(istream& is, SquareMatrix& matrix)
    {
        for (int i = 0; i < matrix.size; ++i) {
            for (int j = 0; j < matrix.size; ++j) {
                is >> matrix.data[i][j];
            }
        }
        return is;
    }


    // Addition operator overload
    SquareMatrix operator+(const SquareMatrix& other)
    {
        if (size != other.size) {
            cout << "Error: the dimensional problem occurred" << endl;
            return {0};
        } else {
            SquareMatrix result(size);
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    result.data[i][j] = data[i][j] + other.data[i][j];
                }
            }
            return result;
        }
    }

    // Subtraction operator overload
    SquareMatrix operator-(const SquareMatrix& other)
    {
        if (size != other.size) {
            cout << "Error: the dimensional problem occurred" << endl;
            return {0};
        } else {
            SquareMatrix result(size);
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    result.data[i][j] = data[i][j] - other.data[i][j];
                }
            }
            return result;
        }
    }

    // Multiplication operator overload
    SquareMatrix operator*(const SquareMatrix& other)
    {
        if (size != other.size) {
            cout << "Error: the dimensional problem occurred" << endl;
            return {0};
        } else {
            SquareMatrix result(size);
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < other.size; ++j) {
                    for (int k = 0; k < size; ++k) {
                        result.data[i][j] += data[i][k] * other.data[k][j];
                    }
                }
            }
            return result;
        }
    }

    // Transpose matrix
    SquareMatrix transpose()
    {
        SquareMatrix result(size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    double determinant() {
        double det = 1.0;
        int counter = 1;
        for (int i = 0; i < size; ++i) {
            // Find the maximum absolute element in the current column
            int maxRow = i;
            for (int j = i + 1; j < size; ++j) {
                if (abs(data[j][i]) > abs(data[maxRow][i])) {
                    maxRow = j;
                }
            }

            // Swap rows if necessary
            if (maxRow != i) {
                swap(data[i], data[maxRow]);
                det *= -1; // Update determinant sign
            }

            // Perform elimination
            for (int j = i + 1; j < size; ++j) {
                double factor = data[j][i] / data[i][i];
                for (int k = i; k < size; ++k) {
                    data[j][k] -= factor * data[i][k];
                }
            }
        }

        // Calculate the determinant
        for (int i = 0; i < size; ++i) {
            det *= data[i][i];
        }

        return det;
    }

    SquareMatrix* inverse() {
        int counter = 1;
        SquareMatrix *result = new SquareMatrix(size);
        for (int i = 0; i < size; ++i) {
            result->data[i][i] = 1.0; // Initialize the result matrix as an identity matrix
        }

        // Augment the matrix with the identity matrix
        vector<vector<double>> augmentedMatrix(size, vector<double>(2 * size, 0.0));
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                augmentedMatrix[i][j] = data[i][j];
                augmentedMatrix[i][j + size] = (i == j) ? 1.0 : 0.0;
            }
        }

        cout << "Augmented matrix:\n";
        for (const auto &row: augmentedMatrix) {
            for (double val: row) {
                std::cout << std::fixed << std::setprecision(2) << val << ' ';
            }
            cout << '\n';
        }

        // Gaussian elimination
        cout << "Gaussian process:\n";
        for (int i = 0; i < size; ++i) {
            // Find the maximum absolute element in the current column
            int maxRow = i;
            for (int j = i + 1; j < size; ++j) {
                if (abs(augmentedMatrix[j][i]) > abs(augmentedMatrix[maxRow][i])) {
                    maxRow = j;
                }
            }

            // Swap rows if necessary
            if (maxRow != i) {

                swap(augmentedMatrix[i], augmentedMatrix[maxRow]);
                cout << "step #" << counter << ": permutation" << endl;
                counter++;
                for (const auto &row: augmentedMatrix) {
                    for (double val: row) {
                        cout << fixed << setprecision(2) << val << ' ';
                    }
                    cout << '\n';
                }
            }

            // Perform elimination
            for (int j = i + 1; j < size; ++j) {
                double factor = augmentedMatrix[j][i] / augmentedMatrix[i][i];
                if (factor == -0){
                    continue;
                }
                for (int k = i; k < 2 * size; ++k) {
                    augmentedMatrix[j][k] -= factor * augmentedMatrix[i][k];
                }
                cout << "step #" << counter << ": elimination" << endl;
                counter++;
                for (const auto &row: augmentedMatrix) {
                    for (double val: row) {
                        cout << fixed << setprecision(2) << val << ' ';
                    }
                    cout << '\n';
                }
            }
        }

        // Backward elimination
        for (int i = size - 1; i >= 0; --i) {
            for (int j = i - 1; j >= 0; --j) {
                double factor = augmentedMatrix[j][i] / augmentedMatrix[i][i];
                if (factor == -0){
                    continue;
                }
                for (int k = i; k < 2 * size; ++k) {
                    augmentedMatrix[j][k] -= factor * augmentedMatrix[i][k];
                }
                cout << "step #" << counter << ": elimination" << endl;
                counter++;
                for (const auto &row: augmentedMatrix) {
                    for (double val: row) {
                        cout << fixed << setprecision(2) << val << ' ';
                    }
                    cout << '\n';
                }
            }
        }

        // Diagonal normalization
        cout << "Diagonal normalization:\n";
        for (int i = 0; i < size; ++i) {
            double factor = augmentedMatrix[i][i];
            for (int j = 0; j < 2 * size; ++j) {
                augmentedMatrix[i][j] /= factor;
            }
        }
        for (const auto &row: augmentedMatrix) {
            for (double val: row) {
                cout << fixed << setprecision(2) << val << ' ';
            }
            cout << '\n';
        }

        // Extract the inverse matrix
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                result->data[i][j] = augmentedMatrix[i][j + size];
            }
        }

        //Printing result
        cout << "Result:\n";
        for (int i = 0; i < result->size; ++i) {
            for (int j = 0; j < result->size; ++j) {
                cout << result->data[i][j] << " ";
            }
            cout << endl;
        }

        return result;
    }

};

class IdentityMatrix : public SquareMatrix {
public:
    IdentityMatrix(int size) : SquareMatrix(size) {
        for (int i = 0; i < size; ++i) {
            data[i][i] = 1;
        }
    }
};

class EliminationMatrix : public SquareMatrix {
public:
    EliminationMatrix(int size, int row, int col, SquareMatrix A) : SquareMatrix(size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (i == row && j == col) {
                    data[i][j] = -A.data[i][j];
                } else if (i == j) {
                    data[i][j] = 1;
                }
            }
        }
    }
};

class PermutationMatrix : public SquareMatrix {
public:
    PermutationMatrix(int size, int row1, int row2) : SquareMatrix(size) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (i == row1) {
                    data[i][j] = (j == row2) ? 1 : 0;
                } else if (i == row2) {
                    data[i][j] = (j == row1) ? 1 : 0;
                } else if (i == j) {
                    data[i][j] = 1;
                }
            }
        }
    }
};

class ColumnVector {
public:
    vector<double> data;

    ColumnVector(size_t size) : data(size, 0.0) {}

    // Overload the + operator for vector addition
    ColumnVector operator+(const ColumnVector& other) const {
        ColumnVector result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    // Overload the * operator for scalar multiplication
    ColumnVector operator*(double scalar) const {
        ColumnVector result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    // Overload the << operator for output
    friend ostream& operator<<(std::ostream& os, const ColumnVector& vec) {
        for (const auto& val : vec.data) {
            os << val << " ";
        }
        return os;
    }

    // Compute the norm (Euclidean length) of the vector
    double norm() const {
        double sum = 0.0;
        for (const auto& val : data) {
            sum += val * val;
        }
        return sqrt(sum);
    }
};


void gaussianElimination(SquareMatrix& A, ColumnVector& b) {
    int n = A.size;
    int counter = 1;

    cout << "Gaussian process:" << endl;
    for (int i = 0; i < n; ++i) {
        // Find pivot
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (abs(A.data[j][i]) > abs(A.data[pivot][i])) {
                pivot = j;
            }
        }

        // Swap rows if necessary
        if (pivot != i) {
            swap(A.data[i], A.data[pivot]);
            swap(b.data[i], b.data[pivot]);

            cout << "step #" << counter << ": permutation" << endl;
            counter++;
            for (const auto &row: A.data) {
                for (double val: row) {
                    cout << fixed << setprecision(2) << val << ' ';
                }
                cout << '\n';
            }
            for (double val: b.data) {
                cout << fixed << setprecision(2) << val << endl;
            }
        }

        // Eliminate other rows
        for (int j = i + 1; j < n; ++j) {

                double factor = A.data[j][i] / A.data[i][i];
                if (abs(factor) == 0){
                    continue;
                }
                for (int k = i; k < n; ++k) {
                    A.data[j][k] -= factor * A.data[i][k];
                }
                b.data[j] -= factor * b.data[i];
                cout << "step #" << counter << ": elimination" << endl;
                counter++;
                for (const auto &row: A.data) {
                    for (double val: row) {
                        cout << fixed << setprecision(2) << val << ' ';
                    }
                    cout << '\n';
                }
                for (double val: b.data) {
                    cout << fixed << setprecision(2) << val << endl;
                }
        }


    }

    // Backward elimination
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            double factor = A.data[j][i] / A.data[i][i];
            if (abs(factor) == 0){
                continue;
            }
            for (int k = i; k < n; ++k) {
                A.data[j][k] -= factor * A.data[i][k];
            }
            b.data[j] -= factor * b.data[i];
            cout << "step #" << counter << ": elimination" << endl;
            counter++;
            for (const auto &row: A.data) {
                for (double val: row) {
                    cout << fixed << setprecision(2) << val << ' ';
                }
                cout << '\n';
            }
            for (double val: b.data) {
                cout << fixed << setprecision(2) << val << endl;
            }
        }
    }

    cout << "Diagonal normalization:\n";
    // Normalize row
    for (int i = 0; i < n; ++i) {
        double factor = A.data[i][i];
        for (int j = i; j < n; ++j) {
            A.data[i][j] /= factor;
        }
        b.data[i] /= factor;
    }
    for (const auto &row: A.data) {
        for (double val: row) {
            cout << fixed << setprecision(2) << val << ' ';
        }
        cout << '\n';
    }
    for (double val: b.data) {
        cout << fixed << setprecision(2) << val << endl;
    }

    // Backward substitution
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            b.data[j] -= A.data[j][i] * b.data[i];
        }
    }
}


int main() {
    int n1;
    cin >> n1;
    SquareMatrix A(n1);
    cin >> A;

    int n2;
    cin >> n2;
    ColumnVector b(n2);
    for (int i = 0; i < n2; ++i) {
        cin >> b.data[i];
    }

    SquareMatrix A2 = A;
    if (A2.determinant() == 0){
        cout << "Error: matrix A is singular" << endl;
        exit(0);
    }

    gaussianElimination(A, b);
    cout << "Result:" << endl;
    for(double val: b.data){
        cout << fixed << setprecision(2) << val << endl;
    }
    return 0;
}