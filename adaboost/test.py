from model import SearchOptimalCut


#print('Small test:')
#searcher = SearchOptimalCut(y=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#                            X=zip([9, 8, 76, 4, 6, 7, 7, -10, -10, -20],
#                                  [-1, -4, -6, 5, 6, 7, 1, 2, 6, 7])
#                            )

#print('-'*100)
y = [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
X = zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                  [1, 3, 2, 4, 5, 6, 7, 8, 9, 13, 11, 12, 10],
                            )
searcher = SearchOptimalCut(y=y, X=X, w=[1]*13, binary_search=False)
test = searcher.search()
print(test)
print('-'*100)
y = [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
X = zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                  [1, 3, 2, 4, 5, 6, 7, 8, 9, 13, 11, 12, 10],
                            )
searcher2 = SearchOptimalCut(y=y, X=X, w=[1]*13, binary_search=True)
test2 = searcher2.search()
print(test2)
#test2 = searcher.search_binary()
#print(test2)