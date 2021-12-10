import numpy as np
from mpi4py import MPI

# communicator and local rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()



import numpy as np
import matplotlib.pyplot as plt
import pickle

np.random.seed(int(rank))
#np.random.seed(0)


class Configuration:
    """Configuration (lattice) of the problem"""

    def __init__(self, population, N, M, beta, T_inf, Ts, Sage, P_initial, FindContactMatrix=False, MCS=0):
        """Constructor"""
        self.M = M  # the linear size of the lattice
        # the vector [1*4] which contains population percentages for each group (then put here the absalute value of number of people in each group)
        self.population = population
        self.N = N  # the total population
        self.beta = beta  # age dependent inverse mobility vector
        self.T_inf = T_inf  # the infective period
        self.Ts = Ts  # the stop time (after Ts the symptomatic phase begins)
        # the age-class susceptibility (probability to get infected when contacted with an infected individual)
        self.Sage = Sage
        # the vector [1*4] which contains the percentages of infected people in each group
        self.P_initial = P_initial
        self.infected = np.zeros(4)  # the number of infected people in each group
        # matrix with i row corresponding to i person's contacts. 1st column corresponds to the first free column index
        self.ContactMatrix = np.zeros((N, 100))
        # AverageContactMatrix[i, j]: the average number of contacts that a single individual of age-group i has with individuals of age-group j
        self.AverageContactMatrix = np.zeros((4, 4))
        self.FindContactMatrix = FindContactMatrix  # if true than code is a bit different
        self.MCS = MCS  # how many MCS should pass to start fill the contact matrix
        self.MCScounter = 0  # how many MCS has been done

        # configuration - 7D matrix.
        # 6 dimensions correspond to the 6D lattice
        # 1 dimension contains information about the age group, if the person infected or not, if the person has antibodies or not, as well as time counter and an index of an individual:
        # 1st layer M[0,:,:,:,:,:] corresponds to population distribution: 0 - the site is free, 1-4 the site is occupied by a person from the corresponding age group
        # 2nd layer M[1,:,:,:,:,:] corresponds to the "Infected" property of an individual. If the site is occupied then it is 1 or 0, otherwise np.nan
        # 3d layer M[2,:,:,:,:,:], likewise, corresponds to the "Antibodies" property of an individual.
        # 4d layer M[3,:,:,:,:,:], the time counter in MCS (Monte Carlo Steps) that is decremented after each MCS. When it is zero an infected person recovers. If a person is healthy then it is np.nan
        # 5d layer M[4,:,:,:,:,:], If a person is healthy then this is np.nan. Otherwise it is the time which is required for person to recover.
        # 6d layer M[5,:,:,:,:,:], If a person is healthy then this is np.nan. Otherwise it is the time after which symptomatic phase begins.
        # 7d layer M[6,:,:,:,:,:], The index of an individual. If there is no person at this site then it is 0
        self.config = np.zeros((7, M, M, M, M, M, M))

        # Put in the "population" vector the number of people in each age group
        self.population[0] = round(self.population[0] * N/100)
        self.population[1] = round(self.population[1] * N/100)
        self.population[2] = round(self.population[2] * N/100)
        self.population[3] = N - (self.population[0] + self.population[1] +
                                  self.population[2])  # to make sum equal to N

        # The auxiliary matrix to make the 1st layer M[0,:,:,:,:,:] of the configuration matrix. Population is randomly distributed
        A = np.hstack([1*np.ones(self.population[0]), 2*np.ones(self.population[1]), 3 *
                       np.ones(self.population[2]), 4*np.ones(self.population[3]), np.zeros(self.M**6 - N)])

        # The auxiliary matrix to make the 2st layer M[1,:,:,:,:,:] of the configuration matrix.
        B = np.hstack([np.ones(round(self.P_initial[0] * self.population[0]/100)), np.zeros(self.population[0] - round(self.P_initial[0] * self.population[0]/100)), np.ones(round(self.P_initial[1] * self.population[1]/100)), np.zeros(self.population[1] - round(self.P_initial[1] * self.population[1]/100)), np.ones(
            round(self.P_initial[2] * self.population[2]/100)), np.zeros(self.population[2] - round(self.P_initial[2] * self.population[2]/100)), np.ones(round(self.P_initial[3] * self.population[3]/100)), np.zeros(self.population[3] - round(self.P_initial[3] * self.population[3]/100)), np.nan * np.ones(self.M**6 - N)])
#         B = np.hstack([np.random.choice([0, 1], size = (self.population[0]), p = [1 - P_initial[0]/100, P_initial[0]/100]), np.random.choice([0, 1], size = (self.population[1]), p = [1 - P_initial[1]/100, P_initial[1]/100]), np.random.choice([0, 1], size = (self.population[2]), p = [1 - P_initial[2]/100, P_initial[2]/100]), np.random.choice([0, 1], size = (self.population[3]), p = [1 - P_initial[3]/100, P_initial[3]/100]), np.nan * np.ones(self.M**6 - N)])

        # The auxiliary matrix to make the 3st layer M[2,:,:,:,:,:] of the configuration matrix. No one initially has antibodies
        C = np.hstack([np.random.choice([0, 1], size=(self.N), p=[1, 0]),
                       np.nan * np.ones(self.M**6 - N)])
        # those who are infective do not have antibodies yet (this line is necessary if we set some people, say 1% to have antibodies)
        C[B == 1] = 0

        # The auxiliary matrix to make the 4st layer M[3,:,:,:,:,:] of the configuration matrix.
        # It contains the time counter (with exponential distribution with a truncation of the distribution at 5*T_inf) wich decreases by 1 each MCS. At time t = 0 the person recovers. If the person is healthy then the value is np.nan
        D = np.hstack([np.random.exponential(size=N, scale=self.T_inf),
                       np.nan*np.ones(self.M**6-N)])
        D[D > 5*self.T_inf] = 5*self.T_inf
        D[B == 0] = np.nan

        # The auxiliary matrix to make the 5st layer M[4,:,:,:,:,:] of the configuration matrix.
        # It does not depend on time (in contrast with M[3,:,:,:,:,:])
        E = D

        # The auxiliary matrix to make the 6st layer M[5,:,:,:,:,:] of the configuration matrix.
        # It contains the time after which symptomatic phase begins. (with exponential distribution with a truncation of the distribution at 5*Ts)
        F = np.hstack([np.random.exponential(size=N, scale=self.Ts), np.nan*np.ones(self.M**6-N)])
        F[F > 5*self.Ts] = 5*self.Ts
        F[B == 0] = np.nan

        # The auxiliary matrix to make the 7st layer M[6,:,:,:,:,:] of the configuration matrix.
        # It contains an index of each person, used for finding the contact matrix
        G = np.hstack([np.arange(1, self.N + 1), np.zeros(self.M**6 - N)])

        # We vstack the auxiliary matrices, shuffle to make random and reshape into 7D matrix
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
        """Calculate the number of infected people in each age group"""
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
                                    b = self.config[6, (i + i1) % self.M, (j + j1) % self.M, (k + k1) % self.M,
                                                    (m + m1) % self.M, (n + n1) % self.M, (l + l1) % self.M]  # neighbour
                                    if b != 0 and b not in self.ContactMatrix[a - 1, 1:]:
                                        self.ContactMatrix[a - 1,
                                                           (int)(self.ContactMatrix[a - 1, 0]) + 1] = b
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

    def _near_neighbors(self, age_group, i, j, k, m, n, l):
        return int((age_group == self.config[0, i, (j + 1) % self.M, k, m, n, l]) +
                   (age_group == self.config[0, i, (j - 1) % self.M, k, m, n, l]) +
                   (age_group == self.config[0, (i + 1) % self.M, j, k, m, n, l]) +
                   (age_group == self.config[0, (i - 1) % self.M, j, k, m, n, l]) +
                   (age_group == self.config[0, i, j, (k + 1) % self.M, m, n, l]) +
                   (age_group == self.config[0, i, j, (k - 1) % self.M, m, n, l]) +
                   (age_group == self.config[0, i, j, k, (m + 1) % self.M, n, l]) +
                   (age_group == self.config[0, i, j, k, (m - 1) % self.M, n, l]) +
                   (age_group == self.config[0, i, j, k, m, (n + 1) % self.M, l]) +
                   (age_group == self.config[0, i, j, k, m, (n - 1) % self.M, l]) +
                   (age_group == self.config[0, i, j, k, m, n, (l + 1) % self.M]) +
                   (age_group == self.config[0, i, j, k, m, n, (l - 1) % self.M]))


def metropolis_move(obj):
    """One elementary step, Metropolis move. It modifies the configuration with the Metropolis algorithm,
    taking into account the adaptive mobility rules"""

    M = obj.M

    # Randomly choose an individual site 1
    i, j, k, m, n, l = np.random.randint(M, size=(6))

    # We exit the function if there is NO person at the randomly selected site
    age_group = obj.config[0, i, j, k, m, n, l]
    if age_group == 0:
        return None

    beta = obj.beta[(int)(age_group - 1)]

    # The first adaptive mobility rule: the infected person stops moving if he goes through symptomatic phase (T < Ts)
    if obj.config[1, i, j, k, m, n, l] == 1 and obj.config[3, i, j, k, m, n, l] < obj.config[4, i, j, k, m, n, l] - obj.config[5, i, j, k, m, n, l]:
        return None

    # randomly choose nearest neighbor destination (site 2)
    i1 = np.random.choice([1, 2, 3, 4, 5, 6])  # randomly choose direction / dimension of movement
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

    # The second adaptive mobility rule: susceptible individuals tend to avoid contacts with the infected ones
    # in their symptomatic phase. If any neighbour at the destination site is going throught the symptomatic phase then we don't do the move.
    if obj.config[1, i, j, k, m, n, l] == 0 and \
        ((obj.config[3][(i+i1+1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i+i1+1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i+i1+1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i+i1-1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i+i1-1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i+i1-1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i+i1) % M, (j+j1+1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i+i1) % M, (j+j1+1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i+i1) % M, (j+j1+1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i+i1) % M, (j+j1-1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i+i1) % M, (j+j1-1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i+i1) % M, (j+j1-1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i+i1) % M, (j+j1) % M, (k + k1+1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i+i1) % M, (j+j1) % M, (k + k1+1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i+i1) % M, (j+j1) % M, (k + k1+1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i+i1) % M, (j+j1) % M, (k + k1-1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i+i1) % M, (j+j1) % M, (k + k1-1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i+i1) % M, (j+j1) % M, (k + k1-1) % M, (m + m1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1+1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1+1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1+1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1-1) % M, (n + n1) % M, (l + l1) % M] < obj.config[4][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1-1) % M, (n + n1) % M, (l + l1) % M] - obj.config[5][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1-1) % M, (n + n1) % M, (l + l1) % M]) or
         (obj.config[3][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1+1) % M, (l + l1) % M] < obj.config[4][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1+1) % M, (l + l1) % M] - obj.config[5][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1+1) % M, (l + l1) % M]) or
         (obj.config[3][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1-1) % M, (l + l1) % M] < obj.config[4][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1-1) % M, (l + l1) % M] - obj.config[5][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1-1) % M, (l + l1) % M]) or
         (obj.config[3][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1+1) % M] < obj.config[4][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1+1) % M] - obj.config[5][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1+1) % M]) or
         (obj.config[3][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1-1) % M] < obj.config[4][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1-1) % M] - obj.config[5][(i+i1) % M, (j+j1) % M, (k + k1) % M, (m + m1) % M, (n + n1) % M, (l + l1-1) % M])):
        return None

    # compute the difference of nearest neighbor effective number for the sites 1 and 2
    delta_N = obj._near_neighbors(age_group, (i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) %
                                  M, (n + n1) % M, (l + l1) % M) - obj._near_neighbors(age_group, i, j, k, m, n, l)

    # accept modification with Metropolis probability
    # if not accepted: leave configuration unchanged
    if np.random.random() < min(1, np.exp(beta*delta_N)):
        obj.config[:, (i + i1) % M, (j + j1) % M, (k + k1) % M, (m + m1) %
                   M, (n + n1) % M, (l + l1) % M] = obj.config[:, i, j, k, m, n, l]
        obj.config[:, i, j, k, m, n, l] = [0, np.nan, np.nan, np.nan, np.nan, np.nan, 0]


def monte_carlo_step(obj):
    """This is a MCS (Monte Carlo Step): we make M**6 steps so that on average every individual
    attempts to move one time. Then we update the epidemiological data of each individual according the SRI model"""

    M = obj.M
    obj.MCScounter += 1
    if obj.FindContactMatrix == True and obj.MCScounter > obj.MCS:
        obj._calculate_ContactMatrix()

    # At the beginning of each timestep, we consider the infected people recover.
    # We go through the lattice and recover certain people whose infection time has come.
    obj.config[3][obj.config[3] > 0] -= 1  # if time is more than zero then subtract 1
    obj.config[4][obj.config[3] <= 0] = np.nan  # if person recovered than T_ind and Ts = np.nan
    obj.config[5][obj.config[3] <= 0] = np.nan

    obj.config[1][obj.config[3] <= 0] = 0  # if time for individual is zero, then he recovers
    obj.config[2][obj.config[3] <= 0] = 1  # and acquires antibodies
    obj.config[3][obj.config[3] <= 0] = np.nan

    for i in range(M**6):
        metropolis_move(obj)

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
                                    obj.config[3, i, j, k, m, n, l] = min(a, 5 * obj.T_inf)
                                    obj.config[4, i, j, k, m, n,
                                               l] = obj.config[3, i, j, k, m, n, l]
                                    a = np.random.exponential(scale=obj.Ts)
                                    obj.config[5, i, j, k, m, n, l] = min(a, 5 * obj.Ts)


# setting the parameters
# the number of MCS correspondinig to one day (as in the paper):
day = 15
# the initial condition infection vector (percantage of infected people in each age group)
P_initial = [0.3, 0.7, 0.1, 0.05]
# population age distribution of Italy
population = [4.8, 9.3, 65.9, 20]
# population of a city simulated
N = 9331
# lenght of the lattice side
M = 6
# infective time (in MCS)
T_inf = 5*day
# stop time (in MCS)
Ts = 1*day
# susceptibility vector (here we set it as in the paper)
Sage = [0.07*1.5, 0.12*1.5, 0.019*1.5, 0.004*1.5]
# the number of steps that we make to ensure the system reaches stationary state
MCSN = 1*day
beta = [3.5, 0, 0.5, 3.5]
FindContactMatrix = False

Ndays = 28  # 3 WEEKS?
infected = np.zeros((Ndays, 4))

c = Configuration([4.8, 9.3, 65.9, 20], N, M, beta, T_inf, Ts,
                  Sage, P_initial, FindContactMatrix, MCSN)
for j in range(Ndays*day):
    if j % day == 0:
        print("simulation. Day ", 1 + j/day,)
        infected[(int)(j/day), :] = c._calculate_infected()
    monte_carlo_step(c)

infected_dic = {'age1': 0, 'age2': 0, 'age3': 0, 'age4': 0}
infected_dic['age1'] = infected[:, 0]
infected_dic['age2'] = infected[:, 1]
infected_dic['age3'] = infected[:, 2]
infected_dic['age4'] = infected[:, 3]

filename = 'infected_numbers_%i'%rank
#filename = 'infected_numbers'
outfile = open(filename, 'wb')
pickle.dump(infected_dic, outfile)
outfile.close()
