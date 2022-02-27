#include<array>
#include<unistd.h>
#include<memory>
#include<fstream>
#include<unordered_map>
#include<unordered_set>
#include<cassert>
#include<chrono>
#include<cstring>
#include<vector>
#include<algorithm>
#include<iostream>
constexpr int HEIGHT = 6;
constexpr int WIDTH = 7;
constexpr int MAX_SCORE = (WIDTH*HEIGHT+1)/2 - 3;
constexpr int MIN_SCORE = -(WIDTH*HEIGHT)/2 + 3;

int64_t mirrow(int64_t key) {
  int64_t res = 0;
  int64_t mask = (1 << (HEIGHT + 1)) - 1;
  for (int i = 0; i < WIDTH; i++) {
    res <<= (HEIGHT + 1);
    res |= key & mask;
    key >>= HEIGHT + 1;
  }
  return res;
}

constexpr int64_t bottom_mask() {
  int64_t res = 0;
  for (int i = 0; i < WIDTH; i++) {
    res += 1LL << (i * (HEIGHT + 1));
  }
  return res;
}

constexpr int64_t full_mask() {
  int64_t res = 0;
  for (int i = 0; i < WIDTH; i++) {
    res += ((1LL << HEIGHT) - 1) << (i * (HEIGHT + 1));
  }
  return res;
}

constexpr int64_t BOTTOM = bottom_mask();
constexpr int64_t FULL = full_mask();

class Table {
public:

  Table(size_t size, int pinThresh): k(size), l(size), h(size), pinThresh(pinThresh) {
    clear();
  }

  Table(int pinThresh): Table((1 << 23) + 9, pinThresh) {}

  void clear() {
    memset(k.data(), 0, k.size() * sizeof(int32_t));
  }

  void put(int64_t key, int8_t low, int8_t hi, int depth) {
    if (depth <= pinThresh) {
      full_cache[key] = std::make_pair(low, hi);
    } else {
      size_t idx = key % k.size();
      k[idx] = (int32_t) key;
      l[idx] = low;
      h[idx] = hi;
    }
  }

  std::pair<int, int> get(int64_t key, int depth) {
    if (depth <= pinThresh) {
      auto it = full_cache.find(key);
      if (it != full_cache.end()) {
        return it->second;
      }
    } else {
      size_t idx = key % k.size();
      if (k[idx] == (int32_t) key) {
        return {l[idx], h[idx]};
      }
    }
    return {MIN_SCORE,MAX_SCORE};
  }

  void load(const std::string& fname) {
    std::ifstream f(fname, std::ifstream::in);
    int64_t key;
    int score;
    while (f >> key >> score) {
      put(key, score, score, 0);
      key = mirrow(key);
      put(key, score, score, 0);
    }
    std::cerr << fname << " loaded" << std::endl;
  }
private:
  const int pinThresh;
  std::unordered_map<int64_t, std::pair<int,int>> full_cache;
  std::vector<int32_t> k;
  std::vector<int8_t> l;
  std::vector<int8_t> h;
};


#define UP(pos, i) (pos << i)
#define DOWN(pos, i) (pos >> i)
#define LEFT(pos, i) (pos >> i * (HEIGHT + 1))
#define RIGHT(pos, i) (pos << i * (HEIGHT + 1))
#define UP_LEFT(pos, i) UP(LEFT(pos, i), i)
#define DOWN_RIGHT(pos, i) DOWN(RIGHT(pos, i), i)
#define UP_RIGHT(pos, i) UP(RIGHT(pos, i), i)
#define DOWN_LEFT(pos, i) DOWN(LEFT(pos, i), i)
int64_t get_winning_moves(int64_t position, int64_t mask) {
  /*
  int64_t res = UP(pos, 1) & UP(pos, 2) & UP(pos, 3);
  res |= LEFT(pos, 1) & LEFT(pos, 2) & LEFT(pos, 3);
  res |= RIGHT(pos, 1) & LEFT(pos, 1) & LEFT(pos, 2);
  res |= RIGHT(pos, 2) & RIGHT(pos, 1) & LEFT(pos, 1);
  res |= RIGHT(pos, 3) & RIGHT(pos, 2) & RIGHT(pos, 1);
  res |= UP_LEFT(pos, 1) & UP_LEFT(pos, 2) & UP_LEFT(pos, 3);
  res |= DOWN_RIGHT(pos, 1) & UP_LEFT(pos, 1) & UP_LEFT(pos, 2);
  res |= DOWN_RIGHT(pos, 2) & DOWN_RIGHT(pos, 1) & UP_LEFT(pos, 1);
  res |= DOWN_RIGHT(pos, 3) & DOWN_RIGHT(pos, 2) & DOWN_RIGHT(pos, 1);
  res |= UP_RIGHT(pos, 1) & UP_RIGHT(pos, 2) & UP_RIGHT(pos, 3);
  res |= DOWN_LEFT(pos, 1) & UP_RIGHT(pos, 1) & UP_RIGHT(pos, 2);
  res |= DOWN_LEFT(pos, 2) & DOWN_LEFT(pos, 1) & UP_RIGHT(pos, 1);
  res |= DOWN_LEFT(pos, 3) & DOWN_LEFT(pos, 2) & DOWN_LEFT(pos, 1);
  return res & (FULL ^ mask);
  */
  uint64_t r = (position << 1) & (position << 2) & (position << 3);

  //horizontal
  uint64_t p = (position << (HEIGHT+1)) & (position << 2*(HEIGHT+1));
  r |= p & (position << 3*(HEIGHT+1));
  r |= p & (position >> (HEIGHT+1));
  p = (position >> (HEIGHT+1)) & (position >> 2*(HEIGHT+1));
  r |= p & (position << (HEIGHT+1));
  r |= p & (position >> 3*(HEIGHT+1));

  //diagonal 1
  p = (position << HEIGHT) & (position << 2*HEIGHT);
  r |= p & (position << 3*HEIGHT);
  r |= p & (position >> HEIGHT);
  p = (position >> HEIGHT) & (position >> 2*HEIGHT);
  r |= p & (position << HEIGHT);
  r |= p & (position >> 3*HEIGHT);

  //diagonal 2
  p = (position << (HEIGHT+2)) & (position << 2*(HEIGHT+2));
  r |= p & (position << 3*(HEIGHT+2));
  r |= p & (position >> (HEIGHT+2));
  p = (position >> (HEIGHT+2)) & (position >> 2*(HEIGHT+2));
  r |= p & (position << (HEIGHT+2));
  r |= p & (position >> 3*(HEIGHT+2));

  return r & (FULL ^ mask);
}

constexpr int64_t get_column_mask(int col) {
  return ((1LL << HEIGHT) - 1) << col * (HEIGHT + 1);
}

constexpr int64_t get_top_mask(int col) {
  return 1LL << (HEIGHT - 1 + col * (HEIGHT + 1));
}

constexpr int64_t get_bottom_mask(int col) {
  return 1LL << (col * (HEIGHT + 1));
}

int count_winning_moves(int64_t pos, int64_t mask) {
  int64_t moves = get_winning_moves(pos, mask);
  int n = 0;
  while (moves) {
    moves &= moves - 1;
    n++;
  }
  return n;
}

class BitBoard {
public:
  BitBoard(int64_t pos = 0, int64_t mask = 0, int moves = 0):pos(pos), mask(mask), moves(moves) {}

  // seq: sequence of chars, each char is a move.
  // '1' -> col 0
  // '2' -> col 1
  // .etc
  BitBoard(const std::string& seq): pos(0), mask(0), moves(0) {
    for (int i = 0; i < seq.length(); i++) {
      int col = seq[i] - '1';
      pos ^= mask;
      mask |= mask + get_bottom_mask(col);
      moves++;
    }
  }

  int64_t key() {
    return pos + mask;
  }

  BitBoard make_move(int col) {
    return BitBoard(pos ^ mask, mask | (mask + (get_bottom_mask(col))), moves + 1);
  }

  int64_t get_legal_moves() {
    return (mask + BOTTOM) & FULL;
  }

  int64_t get_non_losing_moves() {
    int64_t oppo_winning_moves = get_winning_moves(pos ^ mask, mask);
    int64_t legal_moves = get_legal_moves();
    int64_t forced_moves = legal_moves & oppo_winning_moves;
    if (forced_moves) {
      if (forced_moves & (forced_moves - 1)) {
        // more than 1 forced moves
        return 0;
      }
      legal_moves = forced_moves;
    }
    return legal_moves & ~(oppo_winning_moves >> 1);
  }

  bool canWinWithOneMove() {
    return get_winning_moves(pos, mask) & get_legal_moves();
  }

  bool is_winning_move(int col) {
    return get_column_mask(col) & get_winning_moves(pos, mask) & get_legal_moves();
  }

  void sort_move_cols(int* res, int n) {
    std::array<int, WIDTH> score;
    int64_t moves = mask + BOTTOM;
    for (int i = 0; i < n; i++) {
      score[res[i]] = count_winning_moves(pos | (moves & get_column_mask(res[i])), mask);
    }
    for (int i = 1; i < n; i++) {
      int t = res[i];
      int s = score[t];
      int j = i;
      while (j && score[res[j-1]] < s) {
        res[j] = res[j - 1];
        j--;
      }
      res[j] = t;
    }
  }

  void print() {
    for (int i = HEIGHT - 1; i >= 0; i--) {
      for (int j = 0; j < WIDTH; j++) {
        int64_t t = 1LL << ((HEIGHT + 1) * j + i);
        if (mask & t) {
          if ((bool)(pos & t) == (moves % 2 == 0)) {
            std::cout << " X";
          } else {
            std::cout << " O";
          }
        } else {
          std::cout << " -";
        }
      }
      std::cout << std::endl;
    }

    for (int j = 0; j < WIDTH; j++) {
      std::cout << " " << j;
    }
    std::cout << std::endl;
  }

  int64_t pos;
  int64_t mask;
  int moves;
};

class Solver {
public:
  Solver(Table& table): table(table), nodeCount(0) {}

  void reset() {
    nodeCount = 0;
  }

  int negamax(BitBoard a, int alpha, int beta) {
    nodeCount++;
    int64_t moves = a.get_non_losing_moves();
    if (!moves) {
      return -(HEIGHT * WIDTH - a.moves) / 2;
    }
    if (a.moves >= WIDTH * HEIGHT - 2) {
      return 0;
    }
    auto key = a.key();
    auto [low, hi] = table.get(key, a.moves);
    hi = std::min(hi, (HEIGHT * WIDTH - a.moves - 1) / 2);
    low = std::max(low, -(HEIGHT * WIDTH - a.moves - 2) / 2);
    if (low == hi) {
      return low;
    }
    if (low >= beta) {
      return low;
    }
    if (hi <= alpha) {
      return hi;
    }

    alpha = std::max(alpha, low);
    beta = std::min(hi, beta);
    int score = MIN_SCORE;
    int alpha0 = alpha;
    std::array<int, WIDTH> col;
    int n = 0;
    for (int i = 0; i < WIDTH; i++) {
      int idx = WIDTH /  2 + (1 - 2 * (i % 2)) * (i + 1) / 2;
      if (get_column_mask(idx) & moves) {
        col[n++] = idx;
      }
    }
    a.sort_move_cols(col.data(), n);
    for (int i = 0; i < n; i++) {
      int move = col[i];
      BitBoard b = a.make_move(move);
      score =std::max(score, -negamax(b, -beta, -alpha));
      if (score >= beta) {
        break;
      }
      if (score > alpha) {
        alpha = score;
      }
    }
    alpha = alpha0;
    if (score > alpha && score < beta) {
      table.put(key, score, score, a.moves);
    } else if (score <= alpha) {
      table.put(key, low, score, a.moves);
    } else {
      table.put(key, score, hi, a.moves);
    }
    return score;
  }

  int solve(BitBoard b) {
    if (b.canWinWithOneMove()) {
      return 1;
    }
    int min = -(WIDTH * HEIGHT - b.moves) / 2;
    int max = (WIDTH * HEIGHT + 1 - b.moves) / 2;
    while(min < max) {                    // iteratively narrow the min-max exploration window
      int med = min + (max - min)/2;
      if(med <= 0 && min/2 < med) med = min/2;
      else if(med >= 0 && max/2 > med) med = max/2;
      int r = negamax(b, med, med + 1);   // use a null depth window to know if the actual score is greater or smaller than med
      if(r <= med) max = r;
      else min = r;
    }
    return min;
  }

  int64_t nodeCount;
  Table& table;
};

class Searcher {
public:
  Searcher(Solver& solver, int depth): solver(solver), depth(depth) {}

  void search(BitBoard b) {
    if (b.moves <= depth) {
      dfs(b);
    }
  }

private:
  void dfs(BitBoard b) {
    auto key = b.key();
    if (!is_printed(key)){
      std::cout << key << " " << solver.solve(b) << std::endl;
      mark_printed(key);
    }
    if (b.moves >= depth) {
      return;
    }
    auto moves = b.get_non_losing_moves();
    for (int i = 0; i < WIDTH; i++) {
      if (moves & get_column_mask(i)) {
        dfs(b.make_move(i));
      }
    }
  }

  void mark_printed(int64_t key) {
    printed.insert(key);
  }

  bool is_printed(int64_t key) {
    return printed.find(key) != printed.end();
  }

  std::unordered_set<int64_t> printed;
  Solver& solver;
  const int depth;
};


class Agent {
public:
  virtual ~Agent() = default;
  virtual int get_move(BitBoard& b) = 0;
  virtual std::string getName() = 0;
};

class HumanAgent: public Agent {
public:
  virtual int get_move(BitBoard& b) override {
    std::cout << getName() << "> ";
    int move;
    std::cin >> move;
    return move;
  }

  virtual std::string getName() override {
    return "Human";
  }
};

class AI :public Agent{
public:
  AI(Solver& solver, const std::string& name="AI"): Agent(), solver(solver), name(name) {}

  virtual int get_move(BitBoard& b) override {
    std::cout <<  getName() << "> " << std::flush;
    auto moveScores = get_move_scores(b);
    int res;
    if (moveScores.size() == 0) {
      std::cout << ": all moves are losing." << std::endl;
      // return first legal move when losing
      auto moves = b.get_legal_moves();
      for (int i = 0; i < WIDTH; i++) {
        if (get_column_mask(i) & moves) {
          res = i;
          break;
        }
      }
    } else {
      res = -1;
      int max = MIN_SCORE - 1;
      for (auto [move, score] : moveScores) {
        if(score > max) {
          max = score;
          res = move;
        }
        std::cout <<  move << ":" << score << " ";
      }
    }
    std::cout << std::endl;
    std::cout <<  getName() << "> " << res << std::endl;
    return res;
  }

  virtual std::string getName() override {
    return name;
  }

private:
  std::vector<std::pair<int,int>> get_move_scores(BitBoard& b) {
    int64_t moves = b.get_non_losing_moves();
    std::vector<std::pair<int,int>> res;
    for (int i = 0; i < WIDTH; i++) {
      if (moves & get_column_mask(i)) {
        res.emplace_back(i, -solver.solve(b.make_move(i)));
      }
    }
    return res;
  }
  Solver& solver;
  std::string name;
};

void printHelpAndExit() {
  std::cout 
    << " ./main solve  # solve game states, take input from stdin." << std::endl
    << " ./main search [-s <starting-moves>] [-d <depth>] # compute and print score table up to given depth (default 8)" << std::endl
    << " ./main play # play with solver with text UI" << std::endl
    << " Optional flags:"<< std::endl
    << "  -l <score-table-file>  # load score table" << std::endl
    << "  -t <pin-score-depth-thresh> pin score into cache if depth <= this threshold" << std::endl;
  exit(-1);
}

struct Args {
  std::string cmd;
  std::string startingMoves;
  int depth = 8;
  std::string scoreTableFile;
  int pinScoreDepthThreshold = -1;
  bool weakSolver = false;
};

Args parseArgs(int argc, char** argv) {
  Args args;
  if (argc > 1) {
    args.cmd = argv[1];
    for (int i = 2; i < argc; i++) {
      char* s = argv[i];
      if (s[0] == '-') {
        switch (s[1]) {
          case 's':
            args.startingMoves = argv[++i];
            break;
          case 'd':
            args.depth = std::atoi(argv[++i]);
            break;
          case 'l':
            args.scoreTableFile = argv[++i];
            break;
          case 't':
            args.pinScoreDepthThreshold = std::atoi(argv[++i]);
            break;
          default:
            printHelpAndExit();
            break;
        }
      } else {
        printHelpAndExit();
      }
    }
  } else {
    printHelpAndExit();
  }
  return args;
}

void clearScreen() {
  std::cout << "\033[2J\033[1;1H";
}

class GameRunner {
public:
  void play(std::array<std::unique_ptr<Agent>, 2>& agents) {
    BitBoard b;
    int move;
    int turn = 0;
    while (1) { 
      clearScreen();
      b.print();
      int move = agents[turn]->get_move(b);
      if (b.is_winning_move(move)) {
        clearScreen();
        b.make_move(move).print();
        std::cout << agents[turn]->getName() << " wins" << std::endl;
        break;
      }
      b = b.make_move(move);
      turn = 1 - turn;
      sleep(1);
    }
  }
};


int main(int argc, char** argv) {
  Args args = parseArgs(argc, argv);
  Table table(args.pinScoreDepthThreshold);
  if (args.scoreTableFile.size()) {
    table.load(args.scoreTableFile);
  }
  Solver solver(table);
  if (args.cmd == "solve") {
    std::string s;
    int score;
    int testId = 1;
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalDuration = std::chrono::duration<double>::zero();
    while (std::cin >> s >> score) {
      BitBoard b(s);
      solver.reset();
      auto st = std::chrono::high_resolution_clock::now();
      int v = solver.solve(b);
      std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - st;
      totalDuration += duration;
      if (score ==  v) {
        std::cout << "test " << testId++ <<  " pass " << duration.count() << "s node_count " << solver.nodeCount << std::endl;
      } else {
        std::cout << "test " << testId++ <<  " fail: " <<  v  << "!=" << score << std::endl;
        break;
      }
    }
    std::cout << "total duration: " << totalDuration.count() << "s" << std::endl;
  } else if (args.cmd == "search") {
    BitBoard b(args.startingMoves);
    Searcher searcher(solver, args.depth);
    searcher.search(b);
  } else if (args.cmd == "play") {
    BitBoard b;
    int move;
    std::array<std::unique_ptr<Agent>, 2> agents = {
      std::make_unique<AI>(solver, "player-X"),
      std::make_unique<AI>(solver, "player-O")
    };
    GameRunner().play(agents);
  } else {
    printHelpAndExit();
  }
	return 0;
}
