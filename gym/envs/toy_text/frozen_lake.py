import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete
from gym.envs.toy_text.discrete import categorical_sample

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

def make_random_maze(size="8x8", p=0.10, is_slippery=False, goal_rew=1., hole_rew=0., die_on_hole=True):
    """
    p : float (probability of hole)
    """
    def to_s(row, col):
        return row*ncol + col
    def inc(row, col, a):
        if a==0: # left
            col = max(col-1,0)
        elif a==1: # down
            row = min(row+1,nrow-1)
        elif a==2: # right
            col = min(col+1,ncol-1)
        elif a==3: # up
            row = max(row-1,0)
        return (row, col)

    # TODO: Make tunable, why was the map hardcoded like this?
    if size == "8x8":
        nrow, ncol = (8, 8)
    elif size == "4x4":
        nrow, ncol = (4, 4)
    else:
        raise NotImplementedError
        
    P = {s : {a : [] for a in range(4)} for s in range(nrow*ncol)}
    _map = np.random.choice(2, nrow*ncol, replace=True, p=[p, 1-p])
    desc = []
    for i in range(ncol):
        _row = ""
        for j in range(nrow):
            if i == 0 and j == 0:
                _row += 'S'
            elif i == (nrow - 1) and j == (ncol - 1):
                _row += 'G'
            else:
                if _map[i*nrow+j] == 0:
                    _row += 'H'
                else:
                    _row += 'F'
        desc.append(_row)
    desc = np.asarray(desc,dtype='c')
    for row in range(nrow):
        for col in range(ncol):
            s = to_s(row, col)
            for a in range(4):
                li = P[s][a]
                letter = desc[row, col]
                if (letter in b'G') or (letter in b'H' and die_on_hole):
                    li.append((1.0, s, 0, True))
                else:
                    if is_slippery:
                        for b in [(a-1)%4, a, (a+1)%4]:
                            newrow, newcol = inc(row, col, b)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0/3.0, newstate, rew, done))
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        if die_on_hole:
                            done = bytes(newletter) in b'GH'
                        else:
                            done = bytes(newletter) in b'G'
                        rew = float(newletter == b'G') * goal_rew + float(newletter == b'H') * hole_rew
                        li.append((1.0, newstate, rew, done))
    return P, desc

class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=True, goal_rew=1., hole_rew=-0.1, die_on_hole=True):
        self.map_name = map_name
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.goal_rew    = goal_rew
        self.hole_rew    = hole_rew
        self.die_on_hole = die_on_hole 

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if (letter in b'G') or (letter in b'H' and self.die_on_hole):
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            assert False, "Doesn't support optional not dying when you hit the hole"
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                li.append((1.0/3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            if self.die_on_hole:
                                done = bytes(newletter) in b'GH'
                            else:
                                done = bytes(newletter) in b'G'
                            rew = float(newletter == b'G') * self.goal_rew + float(newletter == b'H') * self.hole_rew
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def randomize(self):
        """ Reset the state transition distribution.

        Returns: desc(list): char array of the new map
        for debugging purposes
        """
        self.P, desc = make_random_maze(
            self.map_name, goal_rew=self.goal_rew, hole_rew=self.hole_rew, die_on_hole=self.die_on_hole)
        return desc

    def _reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile
