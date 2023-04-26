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
    """

    def __init__(self, data):
        self.data = data

    @classmethod
    def create(cls, n_players):
        """
        
        """
        allocation = (list(range(n_players)) * 42)[0:42]
        random.shuffle(allocation)
        return cls([Territory(territory_id=t_id, player_id=p_id, armies=1) for t_id, p_id in enumerate(allocation)])

    # ====================== #
    # == Neighbor Methods == #
    # ====================== #   

    def neighbors(self, t_id):
        """
        """
        n_ids = risk.definitions.territory_neighbors[t_id]
        return (t for t in self.data if t.t_id in n_ids)

    def hostile_neighbors(self, t_id):
        """
            
        Args:
            t_id (int): ID of the territory.
                
        Returns:
            generator: Generator of Territories.
        """
        p_id = self.owner(t_id)
        n_ids = risk.definitions.territory_neighbors[t_id]
        return (t for t in self.data if (t.p_id != p_id and t.t_id in n_ids))

    def friendly_neighbors(self, t_id):
        """
        """
        p_id = self.owner(t_id)
        n_ids = risk.definitions.territory_neighbors[t_id]
        return (t for t in self.data if (t.p_id == p_id and t.t_id in n_ids))

    
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
        for t_id in path[1:]:
            val += self.data[t_id].armies
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
        dictionary = {}
        dictionary[source] = [source]
        queue = deque()
        queue.append(source)
        visited = set()
        visited.add(source)

        while queue:
            now = queue.popleft()
            if now == target:
                return True

            for territory in risk.definitions.territory_neighbors[current]:
                if territory in visited or self.owner(territory) != self.owner(source):
                    continue
                new = path[current] + [territory]
                if territory in path and len(temp_dict) >= len(path[territory]):
                    continue
                path[territory] = new
                queue.append(territory)
            visited.add(current)
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

    def continent(self, continent_id):
        """
        Create a generator of all territories that belong to a given continent.
            
        Args:
            continent_id (int): ID of the continent.

        Returns:
            generator: Generator of Territories.
        """
        return (t for t in self.data if t.t_id in risk.definitions.continent_territories[continent_id])

    def n_continents(self, p_id):
        """
        Calculate the total number of continents owned by a player.
        
        Args:
            p_id (int): ID of the player.
                
        Returns:
            int: Number of continents owned by the player.
        """
        return len([continent_id for continent_id in range(6) if self.owns_continent(p_id, continent_id)])

    def owns_continent(self, p_id, continent_id):
        """
        Check if a player owns a continent.
        
        Args:
            p_id (int): ID of the player.
            continent_id (int): ID of the continent.
            
        Returns:
            bool: True if the player owns all of the continent's territories.
        """
        return all((t.p_id == p_id for t in self.continent(continent_id)))

    def continent_owner(self, continent_id):
        """
        Find the owner of all territories in a continent. If the continent
        is owned by various players, return None.
            
        Args:
            continent_id (int): ID of the continent.
                
        Returns:
            int/None: Player_id if a player owns all territories, else None.
        """
        p_ids = set([t.p_id for t in self.continent(continent_id)])
        if len(p_ids) == 1:
            return p_ids.pop()
        return None

    def continent_fraction(self, continent_id, p_id):
        """
        Compute the fraction of a continent a player owns.
        
        Args:
            continent_id (int): ID of the continent.
            p_id (int): ID of the player.

        Returns:
            float: The fraction of the continent owned by the player.
        """
        c_data = list(self.continent(continent_id))
        p_data = [t for t in c_data if t.p_id == p_id]
        return float(len(p_data)) / len(c_data)

    def num_foreign_continent_territories(self, continent_id, p_id):
        """
        Compute the number of territories owned by other players on a given continent.
        
        Args:
            continent_id (int): ID of the continent.
            p_id (int): ID of the player.

        Returns:
            int: The number of territories on the continent owned by other players.
        """
        return sum(1 if t.p_id != p_id else 0 for t in self.continent(continent_id))

    # ==================== #
    # == Action Methods == #
    # ==================== #    

    def reinforcements(self, p_id):
        """
        Calculate the number of reinforcements a player is entitled to.
            
        Args:
            p_id (int): ID of the player.

        Returns:
            int: Number of reinforcement armies that the player is entitled to.
        """
        base_reinforcements = max(3, int(self.n_territories(p_id) / 3))
        bonus_reinforcements = 0
        for continent_id, bonus in risk.definitions.continent_bonuses.items():
            if self.continent_owner(continent_id) == p_id:
                bonus_reinforcements += bonus
        return base_reinforcements + bonus_reinforcements

    def possible_attacks(self, p_id):
        """
        Assemble a list of all possible attacks for the players.

        Args:
            p_id (int): ID of the attacking player.

        Returns:
            list: List of Moves.
        """
        return [Move(from_t.t_id, from_t.armies, to_t.t_id, to_t.p_id, to_t.armies)
                for from_t in self.mobile(p_id) for to_t in self.hostile_neighbors(from_t.t_id)]

    def possible_fortifications(self, p_id):
        """
        Assemble a list of all possible fortifications for the players.
        
        Args:
            p_id (int): ID of the attacking player.

        Returns:
            list: List of Moves.
        """
        return [Move(from_t.t_id, from_t.armies, to_t.t_id, to_t.p_id, to_t.armies)
                for from_t in self.mobile(p_id) for to_t in self.friendly_neighbors(from_t.t_id)]

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
            raise ValueError('Board: Cannot move {n} armies from territory {t_id}.'
                             .format(n=n_armies, t_id=from_territory))
        if to_territory not in [t.t_id for t in self.friendly_neighbors(from_territory)]:
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
            raise ValueError('Board: Cannot attack with {n} armies from territory {t_id}.'
                             .format(n=attackers, t_id=from_territory))
        if to_territory not in [t_id for (t_id, _, _) in self.hostile_neighbors(from_territory)]:
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
            path ([int]): a path of t_ids to plot
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
            self.plot_single(t.t_id, t.p_id, t.armies)

        if not filename:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(filename,bbox_inches='tight')

    @staticmethod
    def plot_single(t_id, p_id, armies):
        """
        Plot a single army dot.
            
        Args:
            t_id (int): the id of the territory to plot,
            p_id (int): the player id of the owner,
            armies (int): the number of armies.
        """
        coor = risk.definitions.territory_locations[t_id]
        plt.scatter(
            [coor[0]*1.2], 
            [coor[1]*1.22], 
            s=400, 
            c=risk.definitions.player_colors[p_id],
            zorder=2
            )
        plt.text(
            coor[0]*1.2, 
            coor[1]*1.22 + 15, 
            s=str(armies),
            color='black' if risk.definitions.player_colors[p_id] in ['yellow', 'pink'] else 'white',
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

    def owner(self, t_id):
        """
        Get the owner of the territory.

        Args:
            t_id (int): ID of the territory.

        Returns:
            int: Player_id that owns the territory.
        """
        return self.data[t_id].p_id

    def armies(self, t_id):
        """
        Get the number of armies on the territory.

        Args:
            t_id (int): ID of the territory.

        Returns:
            int: Number of armies in the territory.
        """
        return self.data[t_id].armies

    def set_owner(self, t_id, p_id):
        """
        Set the owner of the territory.

        Args:
            t_id (int): ID of the territory.
            p_id (int): ID of the player.
        """
        self.data[t_id] = Territory(t_id, p_id, self.armies(t_id))

    def set_armies(self, t_id, n):
        """
        Set the number of armies on the territory.

        Args:
            t_id (int): ID of the territory.
            n (int): Number of armies on the territory.

        Raises:
            ValueError if n < 1.
        """
        if n < 1:
            raise ValueError('Board: cannot set the number of armies to <1 ({t_id}, {n}).'.format(t_id=t_id, n=n))
        self.data[t_id] = Territory(t_id, self.owner(t_id), n)

    def add_armies(self, t_id, n):
        """
        """
        self.set_armies(t_id, self.armies(t_id) + n)

    def n_armies(self, p_id):
        """
        """
        return sum((t.armies for t in self.data if t.p_id == p_id))

    def n_territories(self, p_id):
        """
        """
        return len([None for t in self.data if t.p_id == p_id])

    def territories_of(self, p_id):
        """
        """
        return [t.t_id for t in self.data if t.p_id == p_id]

    def mobile(self, p_id):
        """
        """
        return (t for t in self.data if (t.p_id == p_id and t.armies > 1))
