#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int pixelID(int r, int c, int cols) {
    return r * cols + c;  // flatten 2D to 1D node ID
}


vector<vector<int>> adjacencyList(const vector<vector<int>> mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    int V = rows * cols;

    vector<vector<int>> output(V, vector<int>(V, 0));

    for (int u = 0; u < V; ++u) {
        int r1 = u / cols;
        int c1 = u % cols;

        for (int v = 0; v < V; ++v) {
            int r2 = v / cols;
            int c2 = v % cols;

            output[u][v] = std::abs(mat[r1][c1] - mat[r2][c2]);
        }
    }

    return output;
}



int main() {
    vector<vector<int>> mat = {
        {10, 20, 10},
        {30, 40, 10},
        {30, 40, 10}
    };

    int rows = mat.size();
    int cols = mat[0].size();
    int V = rows * cols;

    for(int i=0; i<V; i++){
        
        cout << "Row :" << i/cols <<" "<<"Col :"<< i % cols << endl;
    }


    return 0;
}
