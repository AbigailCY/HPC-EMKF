#include <iostream>
#include <fstream>
using namespace std;


void writeVector(const char* filename, double* x, int size){
  ofstream myfile (filename);
  if (myfile.is_open())
  {
    myfile << "[";
    for(int count = 0; count < size - 1; count ++){
        myfile << x[count] << ", " ;
    }
    myfile << x[size-1] << "]";
    myfile.close();
  }
  else cout << "Unable to open file";
}
 

int main () {
  const int size = 5;
  double x[] = {1,2,3,4,5}; 
  writeVector("examplePass.txt", x, size); 

  return 0;
}
