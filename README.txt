Instructions to extract from a4.tar.gz and RUN using default inputs as provided in assignment

***Copy and paste these into the directory containing a4.tar.gz***
tar -xzvf a4.tar.gz 
mkdir build
cd build
cmake ..
make
cp ../in.ppm .
cp ../in.pgm .
./5kk70-assignment-gpu

