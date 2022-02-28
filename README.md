
Connect-four game solver inspired by https://blog.gamesolver.org/solving-connect-four/01-introduction/
# God play
![god-play](https://user-images.githubusercontent.com/39423416/155876883-7d9e92e9-a12a-45d9-85f8-5d6385301602.gif)

# Build
```
g++ -std=c++17 -O3 main.cpp
```

# Usage
## Test solver
```
./a.out solve -l table -t 8 < Test_L1_R1
```
## Compute and dump scores for all positions with starting state
```
./a.out search -s 12 -d 8
```

## Play against AI
```
./a.out play -l table -t 8
```

## Watch AI play against itself
```
./a.out play -l table -t 8 -a1 ai -a2 ai
```
