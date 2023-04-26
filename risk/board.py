import os
import random
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

import risk.definitions

from collections import deque
from queue import PriorityQueue
import copy


Territory = namedtuple('Territory', ['territory_id', 'player_id', 'armies'])
Move = namedtuple('Attack', ['from_territory_id', 'from_armies', 'to_territory_id', 'to_player_id', 'to_armies'])


class Board(object):
    """
    The Board object keeps track of all armies situated on the Risk
    world map. Through the definitions it knows the locations of and
    connections between all territories. It handles ownership, attacks
    and movements of armies.
    Args:
        data (list): a sorted list of tuples describing the state of the
            board, each containing three values:
            - tid (int): the territory id of a territory,
            - pid (int): the player id of the owner of the territory,
            - n_armies (int): the number of armies on the territory.
            The list is sorted by the tid, and should be complete.
    """
    def __init__(self, data):
        self.data = data

    @classmethod
    def create(cls, n_players):
        """
        Create a Board and randomly allocate the territories. Place one army on each territory.

        Args:
            n_players (int): Number of players.

        Returns:
            Board: A board with territories randomly allocated to the players.
        """
        allocation = (list(range(n_players)) * 42)[0:42]
        random.shuffle(allocation)
        return cls([Territory(territory_id=tid, player_id=pid, armies=1) for tid, pid in enumerate(allocation)])

    # ====================== #
    # == Neighbor Methods == #
    # ====================== #

    def neighbors(self, territory_id):
        """
        Create a generator of all territories neighboring a given territory.

        Args:
            territory_id (int): ID of the territory to find neighbors of.
        Returns:
            generator: Generator of Territories.
        """
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if t.territory_id in neighbor_ids)

    def hostile_neighbors(self, territory_id):
        """
        Create a generator of all territories neighboring a given territory, of which
        the owner is not the same as the owner of the original territory.

        Args:
            territory_id (int): ID of the territory.

        Returns:
            generator: Generator of Territories.
        """
        player_id = self.owner(territory_id)
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if (t.player_id != player_id and t.territory_id in neighbor_ids))

    def friendly_neighbors(self, territory_id):
        """
        Create a generator of all territories neighboring a given territory, of which
        the owner is the same as the owner of the original territory.
        Args:
            territory_id (int): ID of the territory.
        Returns:
            generator: Generator of tuples of the form (territory_id, player_id, armies).
        """
        player_id = self.owner(territory_id)
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if (t.player_id == player_id and t.territory_id in neighbor_ids))

    
    # ================== #
    # == Path Methods == #
    # ================== #

    def is_valid_path(self, path):
        '''
        '''
        val = True
        if len(path) <= 1:
            return val
        val &= path[0] not in path[1:]
        val &= path[1] in risk.definitions.territory_neighbors[path[0]]
        val &= self.is_valid_path(path[1:])
        return val
    
    def is_valid_attack_path(self, path):
        '''
        '''
        if not self.is_valid_path(path) or len(path) < 2:
            return False
        else:
            for i in range(len(path) - 1):
                if self.owner(path[0]) == self.owner(path[i+1]):
                    return False
        return True


    def cost_of_attack_path(self, path):
        '''
        '''
        if self.is_valid_attack_path(path) is True:
            val = 0
            for tid in path[1:]:
                val += self.data[tid].armies
                return val

    def shortest_path(self, source, target):
        '''
        '''

        dictionary = {}
        dictionary[source] = [source]
        queue = deque()
        queue.append(source)
        visited = set()
        visited.add(source)
        
        while queue:
            now_ter = queue.popleft()
            if now_ter == target:
                return dictionary[now_ter]
            for territory in risk.definitions.territory_neighbors[now_ter]:
                if territory not in visited:
                    dcopy = copy.deepcopy(dictionary[now_ter])
                    dcopy.append(territory)
                    dictionary[territory] = dcopy
                    queue.append(territory)
                visited.add(territory)

    def can_fortify(self, source, target):
        '''
        '''
        owner = self.owner(source)
        if source != self.owner(target):
            return False

        dictionary = {}
        dictionary[source] = [source]
        queue = deque()
        queue.append(source)
        visited = set()
        visited.add(source)

        while queue:
            now_ter = queue.popleft()
            if now_ter == target:
                return True

            move = risk.definitions.territory_neighbors[now_ter]
            for territory in move:
                if territory not in visited and self.owner(territory) == owner:
                    visited.add(territory)
                    pcopy = copy.copy(dictionary[now_ter])
                    pcopy.append(territory)
                    dictionary[territory] = pcopy
                    queue.append(territory)
            return False

    def cheapest_attack_path(self, source, target):
        '''
        '''
        dictionary = {}
        dictionary[source] = [source]
        queue = deque()
        queue.append(source)
        visited = set()
        visited.add(source)
        p_q = PriorityQueue()
        p_q.put((0, source))

        owner = self.owner(source)
        if owner == self.owner(target):
            return False

        while not p_q.empty():
            current_priority, now_ter = p_q.get()
            if now_ter == target:
                return dictionary[now_ter]
            n_ter = risk.definitions.territory_neighbors[now_ter]
            for t in n_ter:
                if t not in visited and owner != self.owner(t):
                    dcopy = dictionary[now_ter].copy()
                    dcopy.append(t)
                    priority = current_priority + self.data[t].armies
                    pr = priority
                    ter = t
                    if ter not in [x[1] for x in p_q.queue]:
                        dictionary[ter] = dcopy
                        p_q.put((priority, ter))
                    elif pr < min([x[0] for x in p_q.queue if x[1] == t]):
                        dictionary[ter] = dcopy
                        for i, (pr, t) in enumerate(p_q.queue):
                            if t == ter:
                                p_q[i] = (priority, ter)
                                break
            visited.add(now_ter) 

    def can_attack(self, source, target):
        '''

        '''

        owner = self.owner(source)
        if owner == self.owner(target):
            return False

        dictionary = {}
        dictionary[source] = [source]
        queue = deque()
        queue.append(source)
        visited = set()
        visited.add(source)

        while queue:
            now_ter = queue.popleft()
            if now_ter == target:
                return True
            n_ter = risk.definitions.territory_neighbors[now_ter]
            for ter in n_ter:
                if ter not in visited and self.owner(ter) != owner:
                    visited.add(ter)
                    dcopy = copy.copy(dictionary[now_ter])
                    dcopy.append(ter)
                    dictionary[ter] = dcopy
                    queue.append(ter)
        return False

    # ======================= #
    # == Continent Methods == #
    # ======================= #

    def continent(self, continentid):
        """
        Create a generator of all territories that belong to a given continent.
            
        Args:
            continentid (int): ID of the continent.

        Returns:
            generator: Generator of Territories.
        """
        return (t for t in self.data if t.tid in risk.definitions.continent_territories[continentid])

    def n_continents(self, pid):
        """
        Calculate the total number of continents owned by a player.
        
        Args:
            pid (int): ID of the player.
                
        Returns:
            int: Number of continents owned by the player.
        """
        return len([continentid for continentid in range(6) if self.owns_continent(pid, continentid)])

    def owns_continent(self, pid, continentid):
        """
        Check if a player owns a continent.
        
        Args:
            pid (int): ID of the player.
            continentid (int): ID of the continent.
            
        Returns:
            bool: True if the player owns all of the continent's territories.
        """
        return all((t.pid == pid for t in self.continent(continentid)))

    def continent_owner(self, continentid):
        """
        Find the owner of all territories in a continent. If the continent
        is owned by various players, return None.
            
        Args:
            continentid (int): ID of the continent.
                
        Returns:
            int/None: Player_id if a player owns all territories, else None.
        """
        pids = set([t.pid for t in self.continent(continentid)])
        if len(pids) == 1:
            return pids.pop()
        return None

    def continent_fraction(self, continentid, pid):
        """
        Compute the fraction of a continent a player owns.
        
        Args:
            continentid (int): ID of the continent.
            pid (int): ID of the player.

        Returns:
            float: The fraction of the continent owned by the player.
        """
        c_data = list(self.continent(continentid))
        p_data = [t for t in c_data if t.pid == pid]
        return float(len(p_data)) / len(c_data)

    def num_foreign_continent_territories(self, continentid, pid):
        """
        Compute the number of territories owned by other players on a given continent.
        
        Args:
            continentid (int): ID of the continent.
            pid (int): ID of the player.

        Returns:
            int: The number of territories on the continent owned by other players.
        """
        return sum(1 if t.pid != pid else 0 for t in self.continent(continentid))

    # ==================== #
    # == Action Methods == #
    # ==================== #    

    def reinforcements(self, pid):
        """
        Calculate the number of reinforcements a player is entitled to.
            
        Args:
            pid (int): ID of the player.

        Returns:
            int: Number of reinforcement armies that the player is entitled to.
        """
        base_reinforcements = max(3, int(self.n_territories(pid) / 3))
        bonus_reinforcements = 0
        for continentid, bonus in risk.definitions.continent_bonuses.items():
            if self.continent_owner(continentid) == pid:
                bonus_reinforcements += bonus
        return base_reinforcements + bonus_reinforcements

    def possible_attacks(self, pid):
        """
        Assemble a list of all possible attacks for the players.

        Args:
            pid (int): ID of the attacking player.

        Returns:
            list: List of Moves.
        """
        return [Move(from_t.tid, from_t.armies, to_t.tid, to_t.pid, to_t.armies)
                for from_t in self.mobile(pid) for to_t in self.hostile_neighbors(from_t.tid)]

    def possible_fortifications(self, pid):
        """
        Assemble a list of all possible fortifications for the players.
        
        Args:
            pid (int): ID of the attacking player.

        Returns:
            list: List of Moves.
        """
        return [Move(from_t.tid, from_t.armies, to_t.tid, to_t.pid, to_t.armies)
                for from_t in self.mobile(pid) for to_t in self.friendly_neighbors(from_t.tid)]

    def fortify(self, from_territory, to_territory, n_armies):
        """
        Perform a fortification.

        Args:
            from_territory (int): Territory_id of the territory where armies leave.
            to_territory (int): Territory_id of the territory where armies arrive.
            n_armies (int): Number of armies to move.

        Raises:
            ValueError if the player moves too many or negative armies.
            ValueError if the territories do not share a border or are not owned by the same player.
        """
        if n_armies < 0 or self.armies(from_territory) <= n_armies:
            raise ValueError('Board: Cannot move {n} armies from territory {tid}.'
                             .format(n=n_armies, tid=from_territory))
        if to_territory not in [t.tid for t in self.friendly_neighbors(from_territory)]:
            raise ValueError('Board: Cannot fortify, territories do not share owner and/or border.')
        self.add_armies(from_territory, -n_armies)
        self.add_armies(to_territory, +n_armies)

    def attack(self, from_territory, to_territory, attackers):
        """
        Perform an attack.

        Args:
            from_territory (int): Territory_id of the offensive territory.
            to_territory (int): Territory_id of the defensive territory.
            attackers (int): Number of attacking armies.

        Raises:
            ValueError if the number of armies is <1 or too large.
            ValueError if a player attacks himself or the territories do not share a border.

        Returns:
            bool: True if the defensive territory was conquered, False otherwise.
        """
        if attackers < 1 or self.armies(from_territory) <= attackers:
            raise ValueError('Board: Cannot attack with {n} armies from territory {tid}.'
                             .format(n=attackers, tid=from_territory))
        if to_territory not in [tid for (tid, _, _) in self.hostile_neighbors(from_territory)]:
            raise ValueError('Board: Cannot attack, territories do not share border or are owned by the same player.')
        defenders = self.armies(to_territory)
        def_wins, att_wins = self.fight(attackers, defenders)
        if self.armies(to_territory) == att_wins:
            self.add_armies(from_territory, -attackers)
            self.set_armies(to_territory, attackers - def_wins)
            self.set_owner(to_territory, self.owner(from_territory))
            return True
        else:
            self.add_armies(from_territory, -def_wins)
            self.add_armies(to_territory, -att_wins)
            return False

    # ====================== #
    # == Plotting Methods == #
    # ====================== #    

    def plot_board(self, path=None, plot_graph=False, filename=None):
        """ 
        Plot the board. 
        
        Args:
            path ([int]): a path of tids to plot
            plot_graph (bool): if true, plots the graph structure overlayed on the board
            filename (str): if given, the plot will be saved to the given filename instead of displayed
        """
        im = plt.imread(os.getcwd() + '/img/risk.png')
        dpi=96
        img_width=800
        fig, ax = plt.subplots(figsize=(img_width/dpi, 300/dpi), dpi=dpi)
        _ = plt.imshow(im)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        def plot_path(xs):
            if not self.is_valid_path(xs):
                print('WARNING: not a valid path')
            coor = risk.definitions.territory_locations[xs[0]]
            verts=[(coor[0]*1.2, coor[1]*1.22 + 25)]
            codes = [ Path.MOVETO ]
            for i,x in enumerate(xs[1:]):
                if (xs[i]==19 and xs[i+1]==1) or (xs[i]==1 and xs[i+1]==19):
                    coor = risk.definitions.territory_locations[x]
                    #verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    verts.append((1000,-200))
                    verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    codes.append(Path.CURVE3)
                    codes.append(Path.CURVE3)
                else:
                    coor = risk.definitions.territory_locations[x]
                    verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    codes.append(Path.LINETO)
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=2)
            ax.add_patch(patch)

        if path is not None:
            plot_path(path)

        if plot_graph:
            for t in risk.definitions.territory_neighbors:
                path = []
                for n in risk.definitions.territory_neighbors[t]:
                    path.append(t)
                    path.append(n)
                plot_path(path)

        for t in self.data:
            self.plot_single(t.tid, t.pid, t.armies)

        if not filename:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(filename,bbox_inches='tight')

    @staticmethod
    def plot_single(tid, pid, armies):
        """
        Plot a single army dot.
            
        Args:
            tid (int): the id of the territory to plot,
            pid (int): the player id of the owner,
            armies (int): the number of armies.
        """
        coor = risk.definitions.territory_locations[tid]
        plt.scatter(
            [coor[0]*1.2], 
            [coor[1]*1.22], 
            s=400, 
            c=risk.definitions.player_colors[pid],
            zorder=2
            )
        plt.text(
            coor[0]*1.2, 
            coor[1]*1.22 + 15, 
            s=str(armies),
            color='black' if risk.definitions.player_colors[pid] in ['yellow', 'pink'] else 'white',
            ha='center',
            size=15
            )

    # ==================== #
    # == Combat Methods == #
    # ==================== #  

    @classmethod
    def fight(cls, attackers, defenders):
        """
        Stage a fight.

        Args:
            attackers (int): Number of attackers.
            defenders (int): Number of defenders.

        Returns:
            tuple (int, int): Number of lost attackers, number of lost defenders.
        """
        n_attack_dices = min(attackers, 3)
        n_defend_dices = min(defenders, 2)
        attack_dices = sorted([cls.throw_dice() for _ in range(n_attack_dices)], reverse=True)
        defend_dices = sorted([cls.throw_dice() for _ in range(n_defend_dices)], reverse=True)
        wins = [att_d > def_d for att_d, def_d in zip(attack_dices, defend_dices)]
        return len([w for w in wins if w is False]), len([w for w in wins if w is True])

    @staticmethod
    def throw_dice():
        """
        """
        return random.randint(1, 6)

    # ======================= #
    # == Territory Methods == #
    # ======================= #

    def owner(self, tid):
        """
        Get the owner of the territory.

        Args:
            tid (int): ID of the territory.

        Returns:
            int: Player_id that owns the territory.
        """
        return self.data[tid].pid

    def armies(self, tid):
        """
        Get the number of armies on the territory.

        Args:
            tid (int): ID of the territory.

        Returns:
            int: Number of armies in the territory.
        """
        return self.data[tid].armies

    def set_owner(self, tid, pid):
        """
        Set the owner of the territory.

        Args:
            tid (int): ID of the territory.
            pid (int): ID of the player.
        """
        self.data[tid] = Territory(tid, pid, self.armies(tid))

    def set_armies(self, tid, n):
        """
        Set the number of armies on the territory.

        Args:
            tid (int): ID of the territory.
            n (int): Number of armies on the territory.

        Raises:
            ValueError if n < 1.
        """
        if n < 1:
            raise ValueError('Board: cannot set the number of armies to <1 ({tid}, {n}).'.format(tid=tid, n=n))
        self.data[tid] = Territory(tid, self.owner(tid), n)

    def add_armies(self, tid, n):
        """
        """
        self.set_armies(tid, self.armies(tid) + n)

    def n_armies(self, pid):
        """
        """
        return sum((t.armies for t in self.data if t.pid == pid))

    def n_territories(self, pid):
        """
        """
        return len([None for t in self.data if t.pid == pid])

    def territories_of(self, pid):
        """
        """
        return [t.tid for t in self.data if t.pid == pid]

    def mobile(self, pid):
        """
        """
        return (t for t in self.data if (t.pid == pid and t.armies > 1))
