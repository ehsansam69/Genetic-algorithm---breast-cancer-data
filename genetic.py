{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import random\n",
    "import matplotlib.pyplot\n",
    "%matplotlib inline "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "cancer=load_breast_cancer()\n",
    "df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])\n",
    "label=cancer[\"target\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 569 entries, 0 to 568\n",
      "Data columns (total 30 columns):\n",
      " #   Column                   Non-Null Count  Dtype  \n",
      "---  ------                   --------------  -----  \n",
      " 0   mean radius              569 non-null    float64\n",
      " 1   mean texture             569 non-null    float64\n",
      " 2   mean perimeter           569 non-null    float64\n",
      " 3   mean area                569 non-null    float64\n",
      " 4   mean smoothness          569 non-null    float64\n",
      " 5   mean compactness         569 non-null    float64\n",
      " 6   mean concavity           569 non-null    float64\n",
      " 7   mean concave points      569 non-null    float64\n",
      " 8   mean symmetry            569 non-null    float64\n",
      " 9   mean fractal dimension   569 non-null    float64\n",
      " 10  radius error             569 non-null    float64\n",
      " 11  texture error            569 non-null    float64\n",
      " 12  perimeter error          569 non-null    float64\n",
      " 13  area error               569 non-null    float64\n",
      " 14  smoothness error         569 non-null    float64\n",
      " 15  compactness error        569 non-null    float64\n",
      " 16  concavity error          569 non-null    float64\n",
      " 17  concave points error     569 non-null    float64\n",
      " 18  symmetry error           569 non-null    float64\n",
      " 19  fractal dimension error  569 non-null    float64\n",
      " 20  worst radius             569 non-null    float64\n",
      " 21  worst texture            569 non-null    float64\n",
      " 22  worst perimeter          569 non-null    float64\n",
      " 23  worst area               569 non-null    float64\n",
      " 24  worst smoothness         569 non-null    float64\n",
      " 25  worst compactness        569 non-null    float64\n",
      " 26  worst concavity          569 non-null    float64\n",
      " 27  worst concave points     569 non-null    float64\n",
      " 28  worst symmetry           569 non-null    float64\n",
      " 29  worst fractal dimension  569 non-null    float64\n",
      "dtypes: float64(30)\n",
      "memory usage: 133.5 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>mean radius</th>\n",
       "      <th>mean texture</th>\n",
       "      <th>mean perimeter</th>\n",
       "      <th>mean area</th>\n",
       "      <th>mean smoothness</th>\n",
       "      <th>mean compactness</th>\n",
       "      <th>mean concavity</th>\n",
       "      <th>mean concave points</th>\n",
       "      <th>mean symmetry</th>\n",
       "      <th>mean fractal dimension</th>\n",
       "      <th>...</th>\n",
       "      <th>worst radius</th>\n",
       "      <th>worst texture</th>\n",
       "      <th>worst perimeter</th>\n",
       "      <th>worst area</th>\n",
       "      <th>worst smoothness</th>\n",
       "      <th>worst compactness</th>\n",
       "      <th>worst concavity</th>\n",
       "      <th>worst concave points</th>\n",
       "      <th>worst symmetry</th>\n",
       "      <th>worst fractal dimension</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>17.99</td>\n",
       "      <td>10.38</td>\n",
       "      <td>122.80</td>\n",
       "      <td>1001.0</td>\n",
       "      <td>0.11840</td>\n",
       "      <td>0.27760</td>\n",
       "      <td>0.3001</td>\n",
       "      <td>0.14710</td>\n",
       "      <td>0.2419</td>\n",
       "      <td>0.07871</td>\n",
       "      <td>...</td>\n",
       "      <td>25.38</td>\n",
       "      <td>17.33</td>\n",
       "      <td>184.60</td>\n",
       "      <td>2019.0</td>\n",
       "      <td>0.1622</td>\n",
       "      <td>0.6656</td>\n",
       "      <td>0.7119</td>\n",
       "      <td>0.2654</td>\n",
       "      <td>0.4601</td>\n",
       "      <td>0.11890</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20.57</td>\n",
       "      <td>17.77</td>\n",
       "      <td>132.90</td>\n",
       "      <td>1326.0</td>\n",
       "      <td>0.08474</td>\n",
       "      <td>0.07864</td>\n",
       "      <td>0.0869</td>\n",
       "      <td>0.07017</td>\n",
       "      <td>0.1812</td>\n",
       "      <td>0.05667</td>\n",
       "      <td>...</td>\n",
       "      <td>24.99</td>\n",
       "      <td>23.41</td>\n",
       "      <td>158.80</td>\n",
       "      <td>1956.0</td>\n",
       "      <td>0.1238</td>\n",
       "      <td>0.1866</td>\n",
       "      <td>0.2416</td>\n",
       "      <td>0.1860</td>\n",
       "      <td>0.2750</td>\n",
       "      <td>0.08902</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>19.69</td>\n",
       "      <td>21.25</td>\n",
       "      <td>130.00</td>\n",
       "      <td>1203.0</td>\n",
       "      <td>0.10960</td>\n",
       "      <td>0.15990</td>\n",
       "      <td>0.1974</td>\n",
       "      <td>0.12790</td>\n",
       "      <td>0.2069</td>\n",
       "      <td>0.05999</td>\n",
       "      <td>...</td>\n",
       "      <td>23.57</td>\n",
       "      <td>25.53</td>\n",
       "      <td>152.50</td>\n",
       "      <td>1709.0</td>\n",
       "      <td>0.1444</td>\n",
       "      <td>0.4245</td>\n",
       "      <td>0.4504</td>\n",
       "      <td>0.2430</td>\n",
       "      <td>0.3613</td>\n",
       "      <td>0.08758</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>11.42</td>\n",
       "      <td>20.38</td>\n",
       "      <td>77.58</td>\n",
       "      <td>386.1</td>\n",
       "      <td>0.14250</td>\n",
       "      <td>0.28390</td>\n",
       "      <td>0.2414</td>\n",
       "      <td>0.10520</td>\n",
       "      <td>0.2597</td>\n",
       "      <td>0.09744</td>\n",
       "      <td>...</td>\n",
       "      <td>14.91</td>\n",
       "      <td>26.50</td>\n",
       "      <td>98.87</td>\n",
       "      <td>567.7</td>\n",
       "      <td>0.2098</td>\n",
       "      <td>0.8663</td>\n",
       "      <td>0.6869</td>\n",
       "      <td>0.2575</td>\n",
       "      <td>0.6638</td>\n",
       "      <td>0.17300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20.29</td>\n",
       "      <td>14.34</td>\n",
       "      <td>135.10</td>\n",
       "      <td>1297.0</td>\n",
       "      <td>0.10030</td>\n",
       "      <td>0.13280</td>\n",
       "      <td>0.1980</td>\n",
       "      <td>0.10430</td>\n",
       "      <td>0.1809</td>\n",
       "      <td>0.05883</td>\n",
       "      <td>...</td>\n",
       "      <td>22.54</td>\n",
       "      <td>16.67</td>\n",
       "      <td>152.20</td>\n",
       "      <td>1575.0</td>\n",
       "      <td>0.1374</td>\n",
       "      <td>0.2050</td>\n",
       "      <td>0.4000</td>\n",
       "      <td>0.1625</td>\n",
       "      <td>0.2364</td>\n",
       "      <td>0.07678</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 30 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
       "0        17.99         10.38          122.80     1001.0          0.11840   \n",
       "1        20.57         17.77          132.90     1326.0          0.08474   \n",
       "2        19.69         21.25          130.00     1203.0          0.10960   \n",
       "3        11.42         20.38           77.58      386.1          0.14250   \n",
       "4        20.29         14.34          135.10     1297.0          0.10030   \n",
       "\n",
       "   mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
       "0           0.27760          0.3001              0.14710         0.2419   \n",
       "1           0.07864          0.0869              0.07017         0.1812   \n",
       "2           0.15990          0.1974              0.12790         0.2069   \n",
       "3           0.28390          0.2414              0.10520         0.2597   \n",
       "4           0.13280          0.1980              0.10430         0.1809   \n",
       "\n",
       "   mean fractal dimension  ...  worst radius  worst texture  worst perimeter  \\\n",
       "0                 0.07871  ...         25.38          17.33           184.60   \n",
       "1                 0.05667  ...         24.99          23.41           158.80   \n",
       "2                 0.05999  ...         23.57          25.53           152.50   \n",
       "3                 0.09744  ...         14.91          26.50            98.87   \n",
       "4                 0.05883  ...         22.54          16.67           152.20   \n",
       "\n",
       "   worst area  worst smoothness  worst compactness  worst concavity  \\\n",
       "0      2019.0            0.1622             0.6656           0.7119   \n",
       "1      1956.0            0.1238             0.1866           0.2416   \n",
       "2      1709.0            0.1444             0.4245           0.4504   \n",
       "3       567.7            0.2098             0.8663           0.6869   \n",
       "4      1575.0            0.1374             0.2050           0.4000   \n",
       "\n",
       "   worst concave points  worst symmetry  worst fractal dimension  \n",
       "0                0.2654          0.4601                  0.11890  \n",
       "1                0.1860          0.2750                  0.08902  \n",
       "2                0.2430          0.3613                  0.08758  \n",
       "3                0.2575          0.6638                  0.17300  \n",
       "4                0.1625          0.2364                  0.07678  \n",
       "\n",
       "[5 rows x 30 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "#splitting the model into training and testing set\n",
    "X_train, X_test, y_train, y_test = train_test_split(df, \n",
    "                                                    label, test_size=0.30, \n",
    "                                                    random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy = 0.9649122807017544\n"
     ]
    }
   ],
   "source": [
    "#training a logistics regression model\n",
    "logmodel = LogisticRegression(max_iter=1200000)\n",
    "logmodel.fit(X_train,y_train)\n",
    "predictions = logmodel.predict(X_test)\n",
    "print(\"Accuracy = \"+ str(accuracy_score(y_test,predictions)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "#defining various steps required for the genetic algorithm\n",
    "def initilization_of_population(size,n_feat):\n",
    "    population = []\n",
    "    for i in range(size):\n",
    "        chromosome = np.ones(n_feat,dtype=np.bool)\n",
    "        chromosome[:int(0.3*n_feat)]=False\n",
    "        np.random.shuffle(chromosome)\n",
    "        population.append(chromosome)\n",
    "    return population"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fitness_score(population):\n",
    "    scores = []\n",
    "    for chromosome in population:\n",
    "        logmodel.fit(X_train.iloc[:,chromosome],y_train)\n",
    "        predictions = logmodel.predict(X_test.iloc[:,chromosome])\n",
    "        scores.append(accuracy_score(y_test,predictions))\n",
    "    scores, population = np.array(scores), np.array(population) \n",
    "    inds = np.argsort(scores)\n",
    "    return list(scores[inds][::-1]), list(population[inds,:][::-1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "def selection(pop_after_fit,n_parents):\n",
    "    population_nextgen = []\n",
    "    for i in range(n_parents):\n",
    "        population_nextgen.append(pop_after_fit[i])\n",
    "    return population_nextgen"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "def crossover(pop_after_sel):\n",
    "    population_nextgen=pop_after_sel\n",
    "    for i in range(len(pop_after_sel)):\n",
    "        child=pop_after_sel[i]\n",
    "        child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]\n",
    "        population_nextgen.append(child)\n",
    "    return population_nextgen"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mutation(pop_after_cross,mutation_rate):\n",
    "    population_nextgen = []\n",
    "    for i in range(0,len(pop_after_cross)):\n",
    "        chromosome = pop_after_cross[i]\n",
    "        for j in range(len(chromosome)):\n",
    "            if random.random() < mutation_rate:\n",
    "                chromosome[j]= not chromosome[j]\n",
    "        population_nextgen.append(chromosome)\n",
    "    #print(population_nextgen)\n",
    "    return population_nextgen\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def generations(size,n_feat,n_parents,mutation_rate,n_gen,X_train,\n",
    "                                   X_test, y_train, y_test):\n",
    "    best_chromo= []\n",
    "    best_score= []\n",
    "    population_nextgen=initilization_of_population(size,n_feat)\n",
    "    for i in range(n_gen):\n",
    "        scores, pop_after_fit = fitness_score(population_nextgen)\n",
    "        print(scores[:2])\n",
    "        pop_after_sel = selection(pop_after_fit,n_parents)\n",
    "        pop_after_cross = crossover(pop_after_sel)\n",
    "        population_nextgen = mutation(pop_after_cross,mutation_rate)\n",
    "        best_chromo.append(pop_after_fit[0])\n",
    "        best_score.append(scores[0])\n",
    "    return best_chromo,best_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.9941520467836257, 0.9941520467836257]\n"
     ]
    }
   ],
   "source": [
    "chromo,score=generations(size=200,n_feat=30,n_parents=100,mutation_rate=0.10,\n",
    "                     n_gen=38,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)\n",
    "logmodel.fit(X_train.iloc[:,chromo[-1]],y_train)\n",
    "predictions = logmodel.predict(X_test.iloc[:,chromo[-1]])\n",
    "print(\"Accuracy score after genetic algorithm is= \"+str(accuracy_score(y_test,predictions)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#with applyin genetic algorithm the accuaracy increased "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
