/*

File: alt_parser.cpp
Brief: Script to extract particles from binary simulation output file
Author: Andrea Klein       <alklein@alumni.stanford.edu>

Example Usage: g++ alt_parser.cpp -o parser
       ./parser > particles.txt
Note: the infile must be in the same directory.

*/

#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

// Change infile here
std::string simfile = "sims/xv_dm.z=00.0000_sim1";

// Total number of particles Np is ~ 2^30
// exact number: 1039119117
int Np = 1073741824;

struct _particle
{
  int ip;
  int ih;
  float x, y, z;
  float vx, vy, vz;
};

int readsim(string fname, _particle *pp) {

  ifstream infile;
  infile.open(fname.c_str(), ios::binary|ios::in);
  double * dummy = new double[1];
  infile.read( (char *)&dummy[0], 8 ); // read first 8 bytes into dummy

  for(int i=0; i<Np; i++)
    {
      // read in current particle
      infile.read( (char *)&pp[i].ip, 8 );
      infile.read( (char *)&pp[i].ih, 8 );
      infile.read( (char *)&pp[i].x, sizeof(float) );
      infile.read( (char *)&pp[i].y, sizeof(float) );
      infile.read( (char *)&pp[i].z, sizeof(float) );
      infile.read( (char *)&pp[i].vx, sizeof(float) );
      infile.read( (char *)&pp[i].vy, sizeof(float) );
      infile.read( (char *)&pp[i].vz, sizeof(float) );

      // immediately print out contents of current particle.
      // currently only printing out positions and velocities
      // cout << pp[i].ip << ' ' 
      // << pp[i].ih << ' '
      cout << pp[i].x  << ' ' << pp[i].y << ' ' << pp[i].z << ' '
	   << pp[i].vx << ' ' << pp[i].vy << ' ' << pp[i].vz << endl;
    }

  return 0;
}

int main() {
  _particle *pp = new _particle[Np];
  readsim(simfile, pp); 
  return 0;
}
