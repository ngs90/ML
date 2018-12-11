import numpy as np
from matplotlib import pyplot as plt
from itertools import product
import math
plt.style.use('seaborn')


class Loss(object):
    """
    Description:
        Calculates the loss function value based on true values, predictions and a specified metric and possibly
        observation weights if relevant for given metric. If no weights given an equal weight is assigned to each observation.
        Available metrics are:
            * 'exponential'
            * 'weighted_error'
            * 'misclassification_error'
    Input:
        :param y: list
        :param y_p: list
        :param metric: str
        :param w: list
    :return: float (indicating the loss value, in general larger values are worse)
    """

    def __init__(self, y, y_p, metric=None, w=None):
        self.y = [float(i) for i in y]
        self.y_pred = [float(i) for i in y_p]
        if w is None:
            self.w = [1 / len(self.y)] * len(self.y)
        else:
            self.w = [float(i) for i in w]
        self.metric = metric

    @property
    def loss_value(self):  # Load metric in scope
        loss = getattr(self, self.metric)  # could i.e. be exponential or weighted error loss
        return loss()

    # Available metrics
    def exponential(self):
        # print([self.y[i] for i in range(len(self.y))])
        return sum(np.exp([self.w[i] * (self.y[i] != self.y_pred[i]) for i in
                           range(len(self.y))]))  # returns sum_i=0^N( exp( weights_i * (y_i == y_pred_i)))

    def weighted_error(self):
        return sum([self.w[i] * (self.y[i] != self.y_pred[i]) for i in range(len(self.y))])

    def misclassification_error(self):
        return sum([1 * (self.y[i] != self.y_pred[i]) for i in range(len(self.y))]) / len(self.y)

    # def __repr__(self):
    #    return self.loss_value()


# Can optimize: Can have a look at: http://www3.stat.sinica.edu.tw/statistica/oldpdf/a7n41.pdf

class SearchOptimalCut(object):
    """
    Description:
        Class to calculate the optimal split point given labels, predictors and optionally weights. If no weighs are specified then equal weight are given to every observation.
    Input:
        y (list)
        X (list of tuples) (or list of lists)
        w (list)
    """

    def __init__(self, y, X, w=None, binary_search=None):
        self.y = y
        self.X = X
        if w is None:
            self.w = [1 / len(y)] * len(y)
        else:
            self.w = w
        self.Xz = list(zip(*X))
        self.loss = float('inf')

        self.base = None
        self.best_predictor_index = None
        self.best_predictor_value = None
        self.best_value_index = None
        self.y_pred = None
        self.binary_search = binary_search

    def search(self):
        #print(self.binary_search)
        if self.binary_search is None:
            #print('Autodetecting search algorithm based on categories:', len(list(set(self.y))))
            # Autodetect if should use binary search or not:
            if len(list(set(self.y))) == 2:
                return self.search_binary()
            else:
                return self.search_multi()
        elif self.binary_search is False:
            return self.search_multi()
        elif self.binary_search is True:
            return self.search_binary()

    def search_binary(self):
        for j, predictor in enumerate(self.Xz):  # looping over the different predictors
            predictor_sorted = sorted(predictor, reverse=False)
            # predictor_sorted_reverse = sorted(predictor, reverse=True)
            sort_mask = np.argsort(predictor)
            #print('Predictor', j)
            #print('Predictor', predictor)
            #print('Predictors', predictor_sorted)
            #print('Sort mask', sort_mask)
            y_sorted = [self.y[i] for i in sort_mask]
            y_sorted_reversed = [yy for yy in list(reversed(y_sorted))]
            #print(y_sorted)
            #print(y_sorted_reversed)
            y_sorted_t = [1 if yy == 0 else 0 for yy in y_sorted]
            y_sorted_reversed_t = [yy for yy in list(reversed(y_sorted_t))]
            w_reversed = [ww for ww in list(reversed(self.w))]

            #print(len(self.w))
            #print(len(y_sorted))
            loss_sorted = np.cumsum(y_sorted*np.asarray(self.w))
            loss_sorted_reversed = np.cumsum(y_sorted_reversed*np.asarray(w_reversed))
            loss_sorted_t = np.cumsum(y_sorted_t*np.asarray(self.w))
            loss_sorted_reversed_t = np.cumsum(y_sorted_reversed_t*np.asarray(w_reversed))


            #print(loss_sorted)
            #print(loss_sorted_reversed)
            #print('#'*50)
            #print(loss_sorted_t)
            #print(loss_sorted_reversed_t)

            loss_sorted_reversed_t_R = list(reversed(loss_sorted_reversed_t))
            loss_sorted_reversed_R = list(reversed(loss_sorted_reversed))

            #print(loss_sorted_reversed_t_R)
            #print(loss_sorted_reversed_R)

            loss_1iflarger = [loss_sorted[i] + (loss_sorted_reversed_t_R+[0])[i+1] for i in range(len(loss_sorted))]
            loss_1ifsmallerorequal = [loss_sorted_t[i] + (loss_sorted_reversed_R+[0])[i+1] for i in range(len(loss_sorted_t))]

            # Now we are ready to calculate the loss function for the optimal split point for the predictor
            if min(loss_1ifsmallerorequal) < min(loss_1iflarger):
                best_value_index_temp = np.argmin(loss_1ifsmallerorequal)
                cut_val_temp = predictor[best_value_index_temp]
                base_temp = '1ifsmallerorequal'
                y_pred_temp = [1 if (k <= cut_val_temp) else 0 for k in predictor]
                loss_temp = Loss(self.y, y_pred_temp, w=self.w, metric='weighted_error').loss_value

            else:
                best_value_index_temp = np.argmin(loss_1iflarger)
                cut_val_temp = predictor[best_value_index_temp]
                base_temp = '1iflarger'
                y_pred_temp = [1 if (k > cut_val_temp) else 0 for k in predictor]
                loss_temp = Loss(self.y, y_pred_temp, w=self.w, metric='weighted_error').loss_value


            #print(best_value_index_temp)
            #print(base_temp)
            #print(loss_1iflarger)
            #print(loss_1ifsmallerorequal)

            if loss_temp < self.loss:
                self.best_predictor_index = j
                self.best_predictor_value = cut_val_temp
                self.best_value_index = best_value_index_temp
                self.base = base_temp
                self.loss = loss_temp
                self.y_pred = y_pred_temp

        return self.y_pred, self.loss, self.best_predictor_value, self.best_value_index, self.best_predictor_index, self.base

    def search_multi(self):
        """
        Main function in the class. Loops over the dataset and finds the optimal splitting point by trying every value
        in the given dataset for each predictor.
        :return:
                y_pred (list)
                loss (float)
                best_predictor_value (float)
                best_value_index (int)
                best_predictor_index (int)
                base (str): can attain two values: "1ifsmallerorequal" or "1iflarger".
        """

        for j, predictor in enumerate(self.Xz):  # looping over the different predictors
            for i, obs in enumerate(predictor):  # looping over observations

                # Cutoff value to check
                cut_val_temp = obs
                # Check the loss function based on the cutoff value

                self.y_pred = [1 if (k <= cut_val_temp) else 0 for k in predictor]
                loss_temp = Loss(self.y, self.y_pred, w=self.w, metric='weighted_error').loss_value

                # Check if loss is smaller than for cut off value that has smallest loss so far.
                if loss_temp < self.loss:
                    self.best_predictor_index = j
                    self.best_predictor_value = cut_val_temp
                    self.best_value_index = i
                    self.base = '1ifsmallerorequal'
                    self.loss = loss_temp

                    # print('Loss:', loss, 'Predictor:', best_predictor_index, 'Value:', best_predictor_value, 'Base:', base)

                # print('Loss:', loss_temp, 'Predictor:', j, 'Value:', cut_val_temp, 'Base:', base)

                self.y_pred = [1 if (k > cut_val_temp) else 0 for k in predictor]
                loss_temp = Loss(self.y, self.y_pred, w=self.w, metric='weighted_error').loss_value
                if loss_temp < self.loss:
                    self.best_predictor_index = j
                    self.best_predictor_value = cut_val_temp
                    self.best_value_index = i
                    self.base = '1iflarger'
                    self.loss = loss_temp

                    # print('Loss:', loss, 'Predictor:', best_predictor_index, 'Value:', best_predictor_value, 'Base:', base)

                # print('Loss:', loss_temp, 'Predictor:', j, 'Value:', cut_val_temp, 'Base:', base)

        if self.base == '1ifsmallerorequal':
            self.y_pred = [1 if k <= self.best_predictor_value else 0 for k in self.Xz[self.best_predictor_index]]
        elif self.base == '1iflarger':
            self.y_pred = [1 if k > self.best_predictor_value else 0 for k in self.Xz[self.best_predictor_index]]

        #print('(Partial) Exponential loss:', self.loss, '(Partial) Misclassification error rate:',
        #      Loss(self.y, self.y_pred, 'misclassification_error').loss_value, 'Predictor:', self.best_predictor_index)

        return self.y_pred, self.loss, self.best_predictor_value, self.best_value_index, self.best_predictor_index, self.base


class Tree(object):

    def __init__(self, y, X, depth=2, w=None, binary_search=None):
        self.y = y
        self.X = X
        self.depth = depth
        if w is None:
            self.w = [1 / len(self.y)] * len(self.y)
        else:
            self.w = [float(i) for i in w]
        self.restrictions = None
        self.decisions = None
        self.Xz = list(zip(*self.X))
        self.binary_search = binary_search

    def create_tree(self):

        # Only include data that belongs to a particular split.
        decision_rules = []
        restrictions = []

        for current_depth in range(self.depth):
            print('Growing tree depth:', current_depth+1, '/', self.depth)

            if current_depth == 0:
                y_in_scope = self.y
                X_in_scope = self.X
                _, _, best_predictor_value, _, best_predictor_index, base = SearchOptimalCut(y=y_in_scope,
                                                                                             X=X_in_scope,
                                                                                             w=self.w,
                                                                                             binary_search=self.binary_search).search()



                restrictions.append({'placement': (0,),
                                     'level': current_depth,
                                     'best_predictor_value': best_predictor_value,
                                     'best_predictor_index': best_predictor_index,
                                     'base': base
                                     })
            else:

                for note in list(product(*[list(range(0, 2))] * (current_depth))):
                    # if (len([i for i in y_in_scope if i == 0]) > 5) & (len([i for i in y_in_scope if i == 1]) > 5):

                    note_in_scope = (0,) + note
                    #print('Handling note:', note_in_scope)
                    relevant_restrictions = []
                    for restriction in restrictions:  # Loop over restriction and check if relevant for this node
                        level = len(restriction['placement'])
                        if restriction['placement'] == note_in_scope[:level]:
                            relevant_restrictions.append(restriction)

                    # For particular node we can now determine the mask
                    masks = []
                    masks_explanations = []
                    for rel_restriction in relevant_restrictions:
                        rel_best_predictor_value = rel_restriction['best_predictor_value']
                        rel_best_predictor_index = rel_restriction['best_predictor_index']
                        if note_in_scope[1 + rel_restriction['level']] == 0:  # lower values (stod -1 før i index)
                            rel_mask = [True if x <= rel_best_predictor_value else False for x in
                                        self.Xz[rel_best_predictor_index]]
                            masks_explanations.append(
                                'Requires predictor {pred} to be smaller than {val}'.format(
                                    pred=rel_best_predictor_index,
                                    val=rel_best_predictor_value))
                        elif note_in_scope[1 + rel_restriction['level']] == 1:  # higher values
                            rel_mask = [True if x > rel_best_predictor_value else False for x in
                                        self.Xz[rel_best_predictor_index]]
                            masks_explanations.append(
                                'Requires predictor {pred} to be larger than {val}'.format(
                                    pred=rel_best_predictor_index,
                                    val=rel_best_predictor_value))
                        masks.append(rel_mask)

                        # print('Masks added', len(masks))

                    # All masks have been collected. Now we filter the dataset.
                    # print(masks)
                    mask = [True if all(i) else False for i in zip(*masks)]

                    # for explanation in masks_explanations:
                    #     print('-----------', explanation)

                    y_in_scope = [self.y[i] for i, im in enumerate(mask) if im is True]
                    X_in_scope = [self.X[i] for i, im in enumerate(mask) if im is True]
                    w_in_scope = [self.w[i] for i, im in enumerate(mask) if im is True]

                    if (len([i for i in y_in_scope if i == 0]) > 5) & (len([i for i in y_in_scope if i == 1]) > 5):
                        _, _, best_predictor_value, _, best_predictor_index, base = SearchOptimalCut(y=y_in_scope,
                                                                                                     X=X_in_scope,
                                                                                                     w=w_in_scope,
                                                                                                     binary_search=self.binary_search).search()
                        restrictions.append({'placement': note_in_scope,
                                             'level': current_depth,
                                             'best_predictor_value': best_predictor_value,
                                             'best_predictor_index': best_predictor_index,
                                             'base': base})
                    else:
                        continue
                        # print('Too few values, will not grow note', note_in_scope)

        self.restrictions = restrictions
        return restrictions

    def traverse_decision(self, x, restrictions, placement=(0,), decision_restriction=None):
        """
        Description:
            Not meant to be called directly from object. Is used by the function "predict".
            Determines decision made by decision tree (which is fully specified from the "restrictions") for one obervation x.
        :return:
        """

        all_restrictions = restrictions

        next_placement = placement
        # print('Placement to look for:', next_placement)

        for restriction_in_scope in all_restrictions:
            # print('Handling restriction placed at:', restriction_in_scope['placement'], restriction_in_scope['placement'], next_placement ,  restriction_in_scope['placement'] == next_placement)
            if restriction_in_scope['placement'] == next_placement:  # then it is the note we are looking for
                prev_placement = next_placement
                # Check if smaller or larger than cutoff value
                # print(restriction_in_scope['best_predictor_index'])
                # print(restriction_in_scope['best_predictor_value'])
                placement_add = 1 if x[restriction_in_scope['best_predictor_index']] > restriction_in_scope[
                    'best_predictor_value'] else 0
                decision_restriction = restriction_in_scope
                next_placement = placement + (placement_add,)
                break
        if placement != next_placement:  # if didn't find more relevant nodes (then we are at the leaf note and a decision can be made)
            # print('Found', placement)
            # print('Next place to look for:', next_placement)
            return self.traverse_decision(x, all_restrictions, placement=next_placement, decision_restriction=decision_restriction)

        # print("Didn't find", placement,' now making a decision:')
        # print(decision_restriction['placement'], decision_restriction['base'] , x[decision_restriction['best_predictor_index']], decision_restriction['best_predictor_value'] )
        if decision_restriction['base'] == '1ifsmallerorequal':
            decision = 1 if x[decision_restriction['best_predictor_index']] <= decision_restriction[
                'best_predictor_value'] else 0
        elif decision_restriction['base'] == '1iflarger':
            decision = 1 if x[decision_restriction['best_predictor_index']] > decision_restriction[
                'best_predictor_value'] else 0

        #self.decision = decision
        return decision

    def predict(self, X=None):
        if X is None:
            X_in_scope = self.X
        else:
            X_in_scope = X
        decisions = []
        for x in X_in_scope:
            decision = self.traverse_decision(x=x, restrictions=self.restrictions, placement=(0,))
            decisions.append(decision)
            # print('Decision is:', decision)

        self.decisions = decisions
        return decisions


class AdaBoost(object):
    def __init__(self, binary_search=None):
        self.tree_data = None
        self.X = None
        self.y = None
        self.trees = None
        self.depth = None
        self.binary_search = binary_search

    def train_model(self, X, y, trees=10, depth=3, w=None):
        self.X = X
        self.y = y
        self.trees = trees
        self.depth = depth

        # Inital weights
        if w is None:
            self.w = [1 / len(y)] * len(y)
        else:
            self.w = w

        tree_data = []
        for tree_id in range(trees):
            print('Creating tree:', tree_id, 'Weights:', self.w)
            # Create tree

            tree = Tree(y=self.y, X=self.X, depth=self.depth, w=self.w, binary_search=self.binary_search)
            restrictions = tree.create_tree()

            # Create predictions with tree
            decisions = tree.predict()

            # Check which predictions are incorrect
            err = Loss(y=self.y, y_p=decisions, w=self.w, metric='weighted_error').loss_value
            # mistakes = ['Mistake' if yt != yp else 'Good' for yt, yp, wc in zip(y,decisions,w)]
            # print('Mistakes',mistakes)
            lerr = math.log((1 - err) / err)  # adjustment rate

            # Update weights (more weight to incorrect predictions)
            self.w = [wc * math.exp(lerr * (yt != yp)) for yt, yp, wc in zip(self.y, decisions, self.w)]
            sumw = sum(self.w)
            # Rescale weights to sum to one
            self.w = [wc / sumw for wc in self.w]
            tree_in_scope = {'id': tree_id, 'restrictions': restrictions, 'decisions': decisions}
            tree_data.append(tree_in_scope)

        self.tree_data = tree_data
        return tree_data

    def predict(self, proba=False, threshold=0.5):
        if self.tree_data is None:
            print('Model not trained. Call adaboost.train_model() to train the model.')
        else:
            for i, tree in enumerate(self.tree_data):
                if i == 0:
                    ndecisions = np.array(tree['decisions'])
                else:
                    ndecisions = ndecisions + np.array(tree['decisions'])
            ndecisions = ndecisions / len(self.tree_data)

            if proba:
                return ndecisions
            else:
                return [1 * (n > threshold) for n in ndecisions]


########################################

def test_data():
    n = 1000
    np.random.seed(seed=0)

    labels = [0] * int(n * 0.5) + [1] * int(n * 0.5)
    x1_0 = list(np.random.normal(loc=-1, scale=0.8, size=int(n * 0.25))) + \
           list(np.random.normal(loc=-1, scale=0.5, size=int(n * 0.15))) + \
           list(np.random.normal(loc=2, scale=0.6, size=int(n * 0.10)))
    x2_0 = list(np.random.normal(loc=1, scale=1, size=int(n * 0.25))) + \
           list(np.random.normal(loc=3, scale=0.5, size=int(n * 0.15))) + \
           list(np.random.normal(loc=-2, scale=0.9, size=int(n * 0.10)))

    x1_1 = list(np.random.normal(loc=0, scale=1, size=int(n * 0.25))) + \
           list(np.random.normal(loc=1, scale=1, size=int(n * 0.25)))
    x2_1 = list(np.random.normal(loc=-1, scale=0.5, size=int(n * 0.25))) + \
           list(np.random.normal(loc=3, scale=0.5, size=int(n * 0.25)))

    x1 = x1_0 + x1_1
    x2 = x2_0 + x2_1

    print(len(x1), len(x2), len(labels))
    mask = np.random.permutation(n)

    # Reorder data
    y, X = [labels[i] for i in mask], list(zip([x1[i] for i in mask], [x2[i] for i in mask]))
    y, X = labels, list(zip(x1, x2))
    w = [1 / len(y)] * len(y)

    return y, X, w, x1_0, x2_0, x1_1, x2_1


if __name__ == '__main__':
    # Setting up test data
    y, X, w, x1_0, x2_0, x1_1, x2_1 = test_data()
    print(y, X, w)

    # Plot original data
    fig, ax = plt.subplots()
    #ax.scatter(x1_0, x2_0, alpha=1, color='C0', s=40, facecolors='none', edgecolors='C0', label='predicted 0',
    #           linewidths=2)
    ax.scatter(x1_0, x2_0, alpha=1, color='C0', s=20, label='true class: 0')
    #ax.scatter(x1_1, x2_1, alpha=1, color='C1', s=40, facecolors='none', edgecolors='C1', label='predicted 1',
    #           linewidths=2)
    ax.scatter(x1_1, x2_1, alpha=1, color='C1', s=20, label='true class: 1')
    ax.legend()
    ax.grid(True)
    #plt.show()

    # Testing Loss class

    # Tests
    print(Loss(y=[1, 0, 1], y_p=[1, 1, 1], metric='exponential').loss_value)
    print(Loss(y=[1, 0, 1], y_p=[1, 1, 1], metric='misclassification_error').loss_value)
    print(Loss(y=[1, 0, 1], y_p=[1, 1, 1], metric='weighted_error').loss_value)
    print(Loss(y=[1, 0, 1], y_p=[1, 1, 1]).weighted_error())

    # Testing search optimal cutoff class

    # Tests
    print('Small test:')
    searcher = SearchOptimalCut(y=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                X=list(zip([9, 8, 76, 4, 6, 7, 7, -10, -10, -20],
                                      [-1, -4, -6, 5, 6, 7, 1, 2, 6, 7]))
                                )
    test = searcher.search()
    print(test)

    print('Test on full dataset:')
    searcher = SearchOptimalCut(y=y, X=X, w=w)
    y_pred_test, loss_test, best_pred_val_est, best_val_index_test, best_pred_index_test, base_test = searcher.search()
    print(loss_test, best_pred_val_est, best_val_index_test, best_pred_index_test, base_test)
    print(searcher.loss)

    # Testing Tree class
    # Tests
    print('Small test:')
    restrictions = Tree(y=[0,0,0,0,0,1,1,1,1,1],
                       X=list(zip([9,8,76,4,6,7,7,-10,-10,-20],[-1,-4,-6,5,6,7,1,2,6,7])),
                       depth=2).create_tree()
    print(restrictions)

    # Bigger test
    print('Bigger test:')
    tree_obj = Tree(y=y, X=X, depth=3)
    restrictions = tree_obj.create_tree()
    print(restrictions)



    # Tests of traverse method of tree
    print((-0.15317123522793763, -0.6608120452142774))
    decision = tree_obj.traverse_decision(x=(-0.15317123522793763, -0.6608120452142774), restrictions=restrictions, placement=(0,))
    print('Decision is:', decision)

    # Test of predict method of tree
    decisions = tree_obj.predict(X=X)
    decisions = tree_obj.predict()
    decisions_test = tree_obj.predict(X=[(-3, -1), (-3, 2), (0, -1), (0, 2)])
    print('X:', [(-3, -1), (-3, 2), (0, -1), (0, 2)])
    print('Decision is:', decisions_test)

    # Plot the outcome from the grown tree
    print('Exponential loss:', Loss(y=y, y_p=decisions, metric='exponential').loss_value)
    print('Misclassification error rate:', Loss(y=y, y_p=decisions, metric='misclassification_error').loss_value)

    # for note in list(combinations_with_replacement(range(0,2), 4)):
    #    print((0,)+note)

    # print(X)
    # print(decisions)
    x1_pred0, x2_pred0 = zip(*[x for i, x in enumerate(X) if decisions[i] == 0])
    # print(x1_pred0, x2_pred0)
    x1_pred1, x2_pred1 = zip(*[x for i, x in enumerate(X) if decisions[i] == 1])

    fig, ax = plt.subplots()
    ax.scatter(x1_pred0, x2_pred0, alpha=1, color='C0', s=40, facecolors='none', edgecolors='C0', label='predicted 0',
               linewidths=2)
    ax.scatter(x1_0, x2_0, alpha=1, color='C0', s=30, label='true class: 0')
    ax.scatter(x1_pred1, x2_pred1, alpha=1, color='C1', s=40, facecolors='none', edgecolors='C1', label='predicted 1',
               linewidths=2)
    ax.scatter(x1_1, x2_1, alpha=1, color='C1', s=30, label='true class: 1')
    ax.legend()
    ax.grid(True)

    max0, max1, min0, min1 = max(list(zip(*X))[0]), max(list(zip(*X))[1]), min(list(zip(*X))[0]), min(list(zip(*X))[1])
    for restriction in restrictions:
        if restriction['best_predictor_index'] == 0:
            plt.plot([restriction['best_predictor_value']] * 2, [min1, max1], color='C4')
        elif restriction['best_predictor_index'] == 1:
            plt.plot([min0, max0], [restriction['best_predictor_value']] * 2, color='C4')

    #plt.show()

    # Testing Adaboost class
    adab = AdaBoost()
    adab.train_model(X=X, y=y, trees=5, depth=5, w=None)
    pred_proba = adab.predict(proba=True)
    pred = adab.predict(proba=False)

    print(pred)
    print(pred_proba)


    # Plotting the adaboost model

    print('Weighted loss:', Loss(y=y, y_p=pred, metric='weighted_error', w=None).loss_value)
    print('Misclassification error rate:', Loss(y=y, y_p=pred, metric='misclassification_error', w=None).loss_value)
    x1_pred0, x2_pred0 = zip(*[x for i, x in enumerate(X) if pred[i] == 0])
    x1_pred1, x2_pred1 = zip(*[x for i, x in enumerate(X) if pred[i] == 1])

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(x1_pred0, x2_pred0, alpha=1, color='C0', s=30, facecolors='none', edgecolors='C0', label='predicted 0',
               linewidths=2)
    ax.scatter(x1_0, x2_0, alpha=1, color='C0', s=20, label='true class: 0')
    ax.scatter(x1_pred1, x2_pred1, alpha=1, color='C1', s=30, facecolors='none', edgecolors='C1', label='predicted 1',
               linewidths=2)
    ax.scatter(x1_1, x2_1, alpha=1, color='C1', s=20, label='true class: 1')
    ax.legend()
    ax.grid(True)
    # plt.figure()
    plt.show()

    # Plotting the subtrees of the adaboost model

    for tr in adab.tree_data:
        decisions = tr['decisions']
        restrictions = tr['restrictions']

        print('Weighted loss:', Loss(y=y, y_p=decisions, metric='weighted_error', w=None).loss_value)
        print('Misclassification error rate:', Loss(y=y, y_p=decisions, metric='misclassification_error', w=None).loss_value)

        # for note in list(combinations_with_replacement(range(0,2), 4)):
        #    print((0,)+note)

        # print(X)
        # print(decisions)
        x1_pred0, x2_pred0 = zip(*[x for i, x in enumerate(X) if decisions[i] == 0])
        # print(x1_pred0, x2_pred0)
        x1_pred1, x2_pred1 = zip(*[x for i, x in enumerate(X) if decisions[i] == 1])

        fig, ax = plt.subplots()
        ax.scatter(x1_pred0, x2_pred0, alpha=1, color='C0', s=40, facecolors='none', edgecolors='C0',
                   label='predicted 0', linewidths=2)
        ax.scatter(x1_0, x2_0, alpha=1, color='C0', s=30, label='true class: 0')
        ax.scatter(x1_pred1, x2_pred1, alpha=1, color='C1', s=40, facecolors='none', edgecolors='C1',
                   label='predicted 1', linewidths=2)
        ax.scatter(x1_1, x2_1, alpha=1, color='C1', s=30, label='true class: 1')
        ax.legend()
        ax.grid(True)

        max0, max1, min0, min1 = max(list(zip(*X))[0]), max(list(zip(*X))[1]), min(list(zip(*X))[0]), min(
            list(zip(*X))[1])
        for restrict in restrictions:
            if restrict['best_predictor_index'] == 0:
                plt.plot([restrict['best_predictor_value']] * 2, [min1, max1], color='C4')
            elif restrict['best_predictor_index'] == 1:
                plt.plot([min0, max0], [restrict['best_predictor_value']] * 2, color='C4')
        plt.show()
        plt.close()

