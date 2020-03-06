"""
Programmer : Zeid Al-Ameedi
Date : 03/02/2020
Collab : Piazza forums, stackoverflow, Dr. Jana Doppa, David Henshaw
Details : We will be using the Item-Item collaborative filtering
algorithm (reccomendation system). Given a dataset of movies compiled of different movies and user ratings,
using similarity score, neighborhood set and ultimately present
top 5 movies in a lexicographic ordering for movies.
I.E. 
User-id1 movie-id1 movie-id2 movie-id3 movie-id4 movie-id5
"""

import operator
import numpy as np
import pandas as pd 
import csv 
from itertools import islice

class IICFilter():
    def __init__(self, infile, outfile="output.txt", n=5, user_top=5):
        self.file = infile
        self.outfile = outfile
        self.neighborhood = n
        self.user_top = user_top
    
    def run(self):
        movie_ratings = self.read_movies()
        max_m_id, max_u_id, m_ids, user_ids = self.unique_columns(movie_ratings)
        n_uniques, norm2 = self.eval_ratings(max_m_id, max_u_id, m_ids, movie_ratings, user_ids)
        scores = self.comp_similarity(n_uniques, m_ids, norm2)
        neighbors = self.neigh_dict(m_ids, scores)
        self.cut_neighborhood(m_ids, neighbors)
        matrix = self.build_matrix(max_m_id, max_u_id, movie_ratings)
        computed = self.neighborhood_ratings(matrix, m_ids, neighbors, user_ids)
        self.isolate_recc(computed, user_ids)

    def eval_ratings(self, max_m_id, max_u_id, m_ids, movie_ratings, user_ids):
        n_uniques = np.zeros((max_m_id + 1, max_u_id + 1))
        for rating in movie_ratings:
            n_uniques[rating[1]][rating[0]] = rating[2]
        self.normalize_matrix(n_uniques, m_ids, user_ids)
        init_matrix = dict()
        for id in m_ids:
            init_matrix[id] = np.linalg.norm(n_uniques[id][:])
        return n_uniques, init_matrix
# here
    def read_movies(self):
        ratings = list()
        f = open(self.file, 'r')
        lines = f.read().splitlines()
        lines.pop(0)
        for line in lines:
            line = line.split(",")
            userid = int(line[0])
            movieid = int(line[1])
            rating = float(line[2])
            ratings.append((userid, movieid, rating))
        f.close()
        return ratings

    def unique_columns(self, ratings):
        userids = set()
        movieids = set()
        for rating in ratings:
            userids.add(rating[0])
            movieids.add(rating[1])
        userids = sorted(userids)
        movieids = sorted(movieids)
        max_userid = max(userids)
        max_movieid = max(movieids)
        return max_movieid, max_userid, movieids, userids

    def normalize_matrix(self, copy, movieids, userids):
        for movieid in movieids:
            num_ratings = 0
            for userid in userids:
                if copy[movieid][userid] != 0:
                    num_ratings += 1
            avg_rating = np.sum(copy[movieid][:]) / num_ratings
            self.reduce_cp(avg_rating, copy, movieid, userids)

    def reduce_cp(self, avg_rating, copy, movieid, userids):
        for userid in userids:
            if copy[movieid][userid] != 0:
                copy[movieid][userid] = copy[movieid][userid] - avg_rating

    def comp_similarity(self, copy, movieids, norm2):
        scores = dict()
        for i in movieids:
            for j in movieids:
                if j > i:
                    dot_product = np.sum(copy[i][:] * copy[j][:])
                    norm2_product = norm2[i] * norm2[j]
                    if norm2_product != 0:
                        scores[(i, j)] = dot_product / norm2_product
                    else:
                        scores[(i, j)] = -1
        return scores

    def neigh_dict(self, movieids, scores):
        neighbors = dict()
        for i in movieids:
            el = list()
            for j in movieids:
                if i < j:
                    el.append((j, scores[(i, j)]))
                elif i > j:
                    el.append((j, scores[(j, i)]))
            neighbors[i] = sorted(el, key=operator.itemgetter(1), reverse=True)
        return neighbors

    def cut_neighborhood(self, movieids, neighbors):
        for movieid in movieids:
            top = list()
            tie = list()
            el = neighbors[movieid]
            if len(el) > self.neighborhood:
                top = self.handle_t(el, tie, top)
            else:
                top = el
            neighbors[movieid] = top

    def handle_t(self, el, tie, top):
        threshold = el[self.neighborhood - 1][1]
        for score in el:
            if score[1] > threshold:
                top.append(score)
            elif score[1] == threshold:
                tie.append(score)
            else:
                break
        tie.sort(key=operator.itemgetter(0))
        top = top + tie[0:(self.neighborhood - len(top))]
        return top

    def build_matrix(self, max_movieid, max_userid, ratings):
        matrix = np.zeros((max_movieid + 1, max_userid + 1))
        for rating in ratings:
            matrix[rating[1]][rating[0]] = rating[2]
        return matrix

    def neighborhood_ratings(self, matrix, movieids, neighbors, userids):
        computed = dict()
        for userid in userids:
            self.neighborhood_helper(computed, matrix, movieids, neighbors, userid)
        return computed

    def neighborhood_helper(self, computed, matrix, movieids, neighbors, userid):
        el = list()
        for movieid in movieids:
            if matrix[movieid][userid] == 0:
                num = 0
                denom = 0
                for n in neighbors[movieid]:
                    if matrix[n[0]][userid] != 0:
                        num += n[1] * matrix[n[0]][userid]
                        denom += n[1]
                if denom > 0:
                    el.append((movieid, num / denom))
        computed[userid] = sorted(el, key=operator.itemgetter(1), reverse=True)

    def isolate_recc(self, computed, userids):
        recs = dict()
        for userid in userids:
            top = list()
            tie = list()
            el = computed[userid]
            if len(el) > self.user_top:
                threshold = el[self.user_top - 1][1]
                self.break_t(el, threshold, tie, top)
                tie.sort(key=operator.itemgetter(0))
                top = top + tie[0:(self.user_top - len(top))]
            else:
                threshold = el[len(el) - 1][1]
                self.tied_ratings(el, threshold, tie, top)
                tie.sort(key=operator.itemgetter(0))
                top = top + tie
            recs[userid] = top
        self.write_results(self.outfile, recs)

    def tied_ratings(self, el, threshold, tie, top):
        for rating in el:
            if rating[1] > threshold:
                top.append(rating)
            else:
                tie.append(rating)

    def break_t(self, el, threshold, tie, top):
        for rating in el:
            if rating[1] > threshold:
                top.append(rating)
            elif rating[1] == threshold:
                tie.append(rating)
            else:
                break

    def write_results(self, outfile, recs):
        f = open(self.outfile, 'w')
        for key, value in sorted(recs.items()):
            f.write(str(key))
            for v in value:
                f.write(' ' + str(v[0]))
            f.write('\n')
        f.close()


def main():
    print("Running. . .")
    iicf = IICFilter("movie-lens-data\\ratings.csv")
    iicf.run()
    print("\nDone.")


if __name__ == '__main__':
    main()