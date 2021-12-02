import numpy as np
import pickle



class Configuration:
    """Configuration (lattice) of the problem"""

    def __init__(self, population, N, M, beta, T_inf, Ts, Sage, FindContactMatrix=False):
        """Constructor"""
        self.M = M  # the length of the lattice in one dimension
        # it's a vector (1*4) which contains population percatages for each group
        self.population = population
        self.N = N  # total population
        self.beta = beta  # age dependent inverse mobility vector
        self.T_inf = T_inf  # infective period
        self.Ts = Ts  # symptomatic phase
        self.Sage = Sage  # specific age-class susceptibility
        # the number of infected people at each group
        self.infected = np.zeros(4)
        # matrix with i row corresponding to i person's contacts.1st column corresponds to the first free column index
        self.ContactMatrix = np.zeros((N, 10000))
        # averge number of contact i with j
        self.AverageContactMatrix = np.zeros((4, 4))
        # if true than code is a bit different
        self.FindContactMatrix = FindContactMatrix

        # config - 7D matrix.
        # 1st layer M[0,:,:] corresponds to population distribution: 0 - the site is free, 1-4 the site is occupied by a parson from corresponding age group
        # 2nd layer M[1,:,:] corresponds to the "Infected" property of an individual. If the site is occupied then it is 1 or 0, otherwise np.nan
        # 3d layer M[2,:,:], likewise, corresponds to the "Antibodies" property of an individual.
        self.config = np.zeros((7, M, M, M, M, M, M))

        self.population[0] = round(self.population[0] * N / 100)
        self.population[1] = round(self.population[1] * N / 100)
        self.population[2] = round(self.population[2] * N / 100)
        # to make sum equal to N
        self.population[3] = N - (self.population[0] +
                                  self.population[1] + self.population[2])

        # create the 1st layer of the config matrix. Population is randomly distributed
        A = np.hstack([1 * np.ones(self.population[0]), 2 * np.ones(self.population[1]), 3 *
                       np.ones(self.population[2]), 4 * np.ones(self.population[3]), np.zeros(self.M**6 - N)])
        # create the 2nd and 3d layer of the config matrix. Population is randomly distributed
        if self.FindContactMatrix == False:
            B = np.hstack([np.random.choice([0, 1], size=(self.population[0]), p=[0.9998, 0.0002]), np.random.choice([0, 1], size=(self.population[1]), p=[0.9993, 0.0007]), np.random.choice(
                [0, 1], size=(self.population[2]), p=[0.9973, 0.0027]), np.random.choice([0, 1], size=(self.population[3]), p=[0.9957, 0.0043]), np.nan * np.ones(self.M**6 - N)])
        else:
            # if we look for beta than everyone is healthy
            B = np.hstack([np.random.choice([0, 1], size=(self.N), p=[
                          1, 0]), np.nan * np.ones(self.M**6 - N)])

        C = np.hstack([np.random.choice([0, 1], size=(self.N), p=[
                      1, 0]), np.nan * np.ones(self.M**6 - N)])
        # those who are infective do not have antibodies yet -> if B[2, i, j] = 1 -> B[3, i, j] = 0
        C[B == 1] = 0

        # Create the 4th layer which contains time counter (with exponential distribution) wich decreases by 1 each MSC. At time t = 0 the person recovers
        D = np.hstack([np.random.exponential(
            size=N, scale=self.T_inf), np.nan * np.ones(self.M**6 - N)])
        D[D > 5 * self.T_inf] = 5 * self.T_inf
        D[B == 0] = np.nan
        # 5th layer contains T_inf. It does not depend on time and = np.nan when person recover
        E = D
        # 6th layer with Ts, also follows exponential distribution
        F = np.hstack([np.random.exponential(
            size=N, scale=self.Ts), np.nan * np.ones(self.M**6 - N)])
        F[F > 5 * self.Ts] = 5 * self.Ts
        F[B == 0] = np.nan

        # G - index of each person, used for finding average number of contacts
        G = np.hstack([np.arange(1, self.N + 1), np.zeros(self.M**6 - N)])

        # then we reshape and shuffle it to make random
        H = np.vstack([A, B, C, D, E, F, G])
        # we transpose because the shuffle function shuffles only along the 1st axis
        H = np.transpose(H)
        np.random.shuffle(H)
        H = np.transpose(H)
        A, B, C, D, E, F, G = H[0], H[1], H[2], H[3], H[4], H[5], H[6]
        self.config = np.stack([A.reshape((M, M, M, M, M, M)), B.reshape((M, M, M, M, M, M)), C.reshape((M, M, M, M, M, M)), D.reshape(
            (M, M, M, M, M, M)), E.reshape((M, M, M, M, M, M)), F.reshape((M, M, M, M, M, M)), G.reshape((M, M, M, M, M, M))])
        self._calculate_infected()

    def _calculate_infected(self):
        """Calculate the number of infected people at each group"""
        for i in range(4):
            self.infected[i] = np.sum(self.config[1][self.config[0] == i + 1])
        return self.infected

    def _calculate_ContactMatrix(self):
        """Calculate the contact matrix"""
        for i in range(self.M):
            for j in range(self.M):
                for k in range(self.M):
                    for m in range(self.M):
                        for n in range(self.M):
                            for l in range(self.M):
                                # 6 layer with index of each person
                                a = (int)(self.config[6, i, j, k, m, n, l])
                                if a == 0:
                                    continue
                                for i1, j1, k1, m1, n1, l1 in [(1, 0, 0, 0, 0, 0), (-1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, -1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, -1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, -1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, -1, 0), (0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, -1)]:
                                    b = self.config[6, (i + i1) % self.M, (j + j1) % self.M, (k + k1) % self.M, (
                                        m + m1) % self.M, (n + n1) % self.M, (l + l1) % self.M]  # neighbour
                                    if b != 0 and b not in self.ContactMatrix[a - 1, 1:]:
                                        self.ContactMatrix[a - 1, (int)(
                                            self.ContactMatrix[a - 1, 0]) + 1] = b
                                        self.ContactMatrix[a - 1, 0] += 1

    def _calculate_AverageContactMatrix(self):
        for i in range(self.population[0]):
            for j in range(1, (int)(self.ContactMatrix[i, 0]) + 1):
                if self.ContactMatrix[i, j] <= self.population[0]:
                    self.AverageContactMatrix[0, 0] += 1
                elif self.ContactMatrix[i, j] <= self.population[0] + self.population[1]:
                    self.AverageContactMatrix[0, 1] += 1
                elif self.ContactMatrix[i, j] <= self.population[0] + self.population[1] + self.population[2]:
                    self.AverageContactMatrix[0, 2] += 1
                elif self.ContactMatrix[i, j] <= self.N:
                    self.AverageContactMatrix[0, 3] += 1
        for i in range(self.population[0], self.population[0] + self.population[1]):
            for j in range(1, (int)(self.ContactMatrix[i, 0]) + 1):
                if self.ContactMatrix[i, j] <= self.population[0]:
                    self.AverageContactMatrix[1, 0] += 1
                elif self.ContactMatrix[i, j] <= self.population[0] + self.population[1]:
                    self.AverageContactMatrix[1, 1] += 1
                elif self.ContactMatrix[i, j] <= self.population[0] + self.population[1] + self.population[2]:
                    self.AverageContactMatrix[1, 2] += 1
                elif self.ContactMatrix[i, j] <= self.N:
                    self.AverageContactMatrix[1, 3] += 1
        for i in range(self.population[0] + self.population[1], self.population[0] + self.population[1] + self.population[2]):
            for j in range(1, (int)(self.ContactMatrix[i, 0]) + 1):
                if self.ContactMatrix[i, j] <= self.population[0]:
                    self.AverageContactMatrix[2, 0] += 1
                elif self.ContactMatrix[i, j] <= self.population[0] + self.population[1]:
                    self.AverageContactMatrix[2, 1] += 1
                elif self.ContactMatrix[i, j] <= self.population[0] + self.population[1] + self.population[2]:
                    self.AverageContactMatrix[2, 2] += 1
                elif self.ContactMatrix[i, j] <= self.N:
                    self.AverageContactMatrix[2, 3] += 1
        for i in range(self.population[0] + self.population[1] + self.population[2], self.N):
            for j in range(1, (int)(self.ContactMatrix[i, 0]) + 1):
                if self.ContactMatrix[i, j] <= self.population[0]:
                    self.AverageContactMatrix[3, 0] += 1
                elif self.ContactMatrix[i, j] <= self.population[0] + self.population[1]:
                    self.AverageContactMatrix[3, 1] += 1
                elif self.ContactMatrix[i, j] <= self.population[0] + self.population[1] + self.population[2]:
                    self.AverageContactMatrix[3, 2] += 1
                elif self.ContactMatrix[i, j] <= self.N:
                    self.AverageContactMatrix[3, 3] += 1
        self.AverageContactMatrix[0, :] /= self.population[0]
        self.AverageContactMatrix[1, :] /= self.population[1]
        self.AverageContactMatrix[2, :] /= self.population[2]
        self.AverageContactMatrix[3, :] /= self.population[3]

    def _near_neighbors(self, i, j, k, m, n, l):
        return int(self.config[0, i, j, k, m, n, l] == self.config[0, i, (j + 1) % self.M, k, m, n, l]) + \
            (self.config[0, i, j, k, m, n, l] == self.config[0, i, (j - 1) % self.M, k, m, n, l]) + \
            (self.config[0, i, j, k, m, n, l] == self.config[0, (i + 1) % self.M, j, k, m, n, l]) + \
            (self.config[0, i, j, k, m, n, l] == self.config[0, (i - 1) % self.M, j, k, m, n, l]) + \
            (self.config[0, i, j, k, m, n, l] == self.config[0, i, j, (k + 1) % self.M, m, n, l]) + \
            (self.config[0, i, j, k, m, n, l] == self.config[0, i, j, (k - 1) % self.M, m, n, l] +
             self.config[0, i, j, k, m, n, l] == self.config[0, i, j, k, (m + 1) % self.M, n, l] +
             self.config[0, i, j, k, m, n, l] == self.config[0, i, j, k, (m - 1) % self.M, n, l] +
             self.config[0, i, j, k, m, n, l] == self.config[0, i, j, k, m, (n + 1) % self.M, l] +
             self.config[0, i, j, k, m, n, l] == self.config[0, i, j, k, m, (n - 1) % self.M, l] +
             self.config[0, i, j, k, m, n, l] == self.config[0, i, j, k, m, n, (l + 1) % self.M] +
             self.config[0, i, j, k, m, n, l] == self.config[0, i, j, k, m, n, (l - 1) % self.M])


def metropolis_move1(obj):
    """One step, modify (or not) the configuration with Metropolis algorithm"""

    M = obj.M

    # randomly choose an individual located at site 1
    i, j, k, m, n, l = np.random.randint(M, size=(6))

    # we exit the function if there is NO person at the randomly selected site
    age_group = obj.config[0, i, j, k, m, n, l]
    if age_group == 0:
        return None

    beta = obj.beta[(int)(age_group - 1)]

    # The first adaptive rule: the infected person stops moving if he goes through symptomatic phase (T < Ts)
    if obj.config[1, i, j, k, m, n, l] == 1 and obj.config[3, i, j, k, m, n, l] < obj.config[4, i, j, k, m, n, l] - obj.config[5, i, j, k, m, n, l]:
        return None

    # randomly choose nearest neighbor destination (site 2)
    # randomly choose direction / dimension of movement
    i1 = np.random.choice([1, 2, 3, 4, 5, 6])
    j1 = np.random.choice([1, -1])  # forward or backward
    if i1 == 1:
        i1, j1, k1, m1, n1, l1 = j1, 0, 0, 0, 0, 0
    if i1 == 2:
        i1, j1, k1, m1, n1, l1 = 0, j1, 0, 0, 0, 0
    if i1 == 3:
        i1, j1, k1, m1, n1, l1 = 0, 0, j1, 0, 0, 0
    if i1 == 4:
        i1, j1, k1, m1, n1, l1 = 0, 0, 0, j1, 0, 0
    if i1 == 5:
        i1, j1, k1, m1, n1, l1 = 0, 0, 0, 0, j1, 0
    if i1 == 6:
        i1, j1, k1, m1, n1, l1 = 0, 0, 0, 0, 0, j1

    # Exit the function if there IS person at the randomly selected destination site 2
    if obj.config[0, (i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] != 0:
        return None

    # The second adaptive rule: susceptible individuals tend to avoid contacts with the infected ones
    # in their symptomatic phase
    if obj.config[1, i, j, k, m, n, l] == 0 and \
        ((obj.config[3][(i + i1 + 1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i + i1 + 1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i + i1 + 1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i + i1 - 1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i + i1 - 1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i + i1 - 1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i + i1) % M, (j + j1 + 1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i + i1) % M, (j + j1 + 1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i + i1) % M, (j + j1 + 1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i + i1) % M, (j + j1 - 1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i + i1) % M, (j + j1 - 1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i + i1) % M, (j + j1 - 1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i + i1) % M, (j + j1) % M, (k + k1 + 1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i + i1) % M, (j + j1) % M, (k + k1 + 1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i + i1) % M, (j + j1) % M, (k + k1 + 1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i + i1) % M, (j + j1) % M, (k + k1 - 1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i + i1) % M, (j + j1) % M, (k + k1 - 1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i + i1) % M, (j + j1) % M, (k + k1 - 1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1 + 1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1 + 1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1 + 1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1 - 1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1 - 1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1 - 1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1 + 1) % M, (l + l1) % M] < obj.config[4][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1 + 1) % M, (l + l1) % M] - obj.config[5][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1 + 1) % M, (l + l1) % M]) or
         (obj.config[3][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1 - 1) % M, (l + l1) % M] < obj.config[4][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1 - 1) % M, (l + l1) % M] - obj.config[5][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1 - 1) % M, (l + l1) % M]) or
         (obj.config[3][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1 + 1) % M] < obj.config[4][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1 + 1) % M] - obj.config[5][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1 + 1) % M]) or
         (obj.config[3][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1 - 1) % M] < obj.config[4][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1 - 1) % M] - obj.config[5][(i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1 - 1) % M])):
        return None

    # compute the difference of nearest neighbor effective number for the sites 1 and 2
    delta_N = obj._near_neighbors((i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) %
                                  M, (n + n1) % M, (l + l1) % M) - obj._near_neighbors(i, j, k, m, n, l)

    # accept modification with Metropolis probability
    # if not accepted: leave configuration unchanged
    if np.random.random() < min(1, np.exp(-beta * delta_N)):
        obj.config[:, (i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) %
                   M, (n + n1) % M, (l + l1) % M] = obj.config[:, i, j, k, m, n, l]
        obj.config[:, i, j, k, m, n, l] = [
            0, np.nan, np.nan, np.nan, np.nan, np.nan, 0]


def metropolis_move(obj):
    """This is MCS (Monte Carlo Step): we make M**6 steps so that on average every individual
    attempts to move one time per MCS"""

    M = obj.M
    if obj.FindContactMatrix == True:
        obj._calculate_ContactMatrix()

    # At the beginning of each timestep, we consider the infected people recover.
    # We go through the lattice and recover certain people whose infection time has come.
    # if time is more than zero then subtract 1
    obj.config[3][obj.config[3] > 0] -= 1
    # if person recovered than T_ind and Ts = np.nan
    obj.config[4][obj.config[3] <= 0] = np.nan
    obj.config[5][obj.config[3] <= 0] = np.nan

    # if time for individual is zero, then he recovers
    obj.config[1][obj.config[3] <= 0] = 0
    obj.config[2][obj.config[3] <= 0] = 1  # and acquires antibodies
    obj.config[3][obj.config[3] <= 0] = np.nan

    for i in range(M**6):
        metropolis_move1(obj)

    # if we look for the contact matrix then everyone is healthy and no need to renew the matrices of infection/antibodies etc.
    if obj.FindContactMatrix == False:
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    for m in range(M):
                        for n in range(M):
                            for l in range(M):
                                age_group = obj.config[0, i, j, k, m, n, l]
                                if age_group == 0:
                                    continue

                                # if the selected person does not have antibodies and not infected and some of the new neighbors are infected
                                # then the person gets infected with probability Sage[age group]
                                # Mb better to create new matrix (?)
                                if np.random.random() < obj.Sage[int(age_group - 1)] and obj.config[1, i, j, k, m, n, l] == 0 and \
                                    obj.config[2, i, j, k, m, n, l] == 0 and \
                                    (obj.config[1, (i + 1) % M, j, k, m, n, l] == 1 or obj.config[1, i - 1, j, k, m, n, l] == 1 or
                                     obj.config[1, i, (j + 1) % M, k, m, n, l] == 1 or obj.config[1, i, j - 1, k, m, n, l] == 1 or
                                     obj.config[1, i, j, (k + 1) % M, m, n, l] == 1 or obj.config[1, i, j, k - 1, m, n, l] == 1 or
                                     obj.config[1, i, j, k, (m + 1) % M, n, l] == 1 or obj.config[1, i, j, k, m - 1, n, l] == 1 or
                                     obj.config[1, i, j, k, m, (n + 1) % M, l] == 1 or obj.config[1, i, j, k, m, n - 1, l] == 1 or
                                     obj.config[1, i, j, k, m, n, (l + 1) % M] == 1 or obj.config[1, i, j, k, m, n, l - 1] == 1):
                                    obj.config[1, i, j, k, m, n, l] = 1
                                    a = np.random.exponential(scale=obj.T_inf)
                                    obj.config[3, i, j, k, m, n, l] = min(
                                        a, 5 * obj.T_inf)
                                    obj.config[4, i, j, k, m, n,
                                               l] = obj.config[3, i, j, k, m, n, l]
                                    a = np.random.exponential(scale=obj.Ts)
                                    obj.config[5, i, j, k, m, n, l] = min(
                                        a, 5 * obj.Ts)


day = 15
# all the data from the paper. Population of the city (23530) and the length of the lattice (7) are chosen so that population density is equal to 20%
FindContactMatrix = True
c = Configuration([4.8, 9.3, 65.9, 20], 23530, 7, [0.50, 0.01, 0.085, 0.95],
                  5 * day, 1 * day, [0.07, 0.12, 0.0019, 0.004], FindContactMatrix)
for i in range(1 * day):
    metropolis_move(c)
    infected = np.vstack([infected, c._calculate_infected()])

infected_dic = {'age1': 0, 'age2': 0, 'age3': 0, 'age4': 0}
infected_dic['age1'] = infected[:, 0] * 1000 / c.population[0]
infected_dic['age2'] = infected[:, 1] * 1000 / c.population[0]
infected_dic['age3'] = infected[:, 2] * 1000 / c.population[0]
infected_dic['age4'] = infected[:, 3] * 1000 / c.population[0]

filename = 'infected_numbers'
outfile = open(filename, 'wb')
pickle.dump(infected_dic, outfile)
outfile.close()


# infected = infected[1:,:]
