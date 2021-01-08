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
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.9941520467836257, 0.9941520467836257]\n",
      "[0.9883040935672515, 0.9883040935672515]\n",
      "[0.9941520467836257, 0.9941520467836257]\n",
      "[0.9941520467836257, 0.9941520467836257]\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-31-56fcd81e9aed>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m chromo,score=generations(size=200,n_feat=30,n_parents=100,mutation_rate=0.10,\n\u001b[1;32m----> 2\u001b[1;33m                      n_gen=38,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0mlogmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mchromo\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m-\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my_train\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mpredictions\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlogmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_test\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mchromo\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m-\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Accuracy score after genetic algorithm is= \"\u001b[0m\u001b[1;33m+\u001b[0m\u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0maccuracy_score\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mpredictions\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-26-c1c0d2618b87>\u001b[0m in \u001b[0;36mgenerations\u001b[1;34m(size, n_feat, n_parents, mutation_rate, n_gen, X_train, X_test, y_train, y_test)\u001b[0m\n\u001b[0;32m      5\u001b[0m     \u001b[0mpopulation_nextgen\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0minitilization_of_population\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msize\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mn_feat\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      6\u001b[0m     \u001b[1;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mn_gen\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 7\u001b[1;33m         \u001b[0mscores\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpop_after_fit\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfitness_score\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mpopulation_nextgen\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      8\u001b[0m         \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mscores\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      9\u001b[0m         \u001b[0mpop_after_sel\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mselection\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mpop_after_fit\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mn_parents\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-19-4753d3c24cd8>\u001b[0m in \u001b[0;36mfitness_score\u001b[1;34m(population)\u001b[0m\n\u001b[0;32m      2\u001b[0m     \u001b[0mscores\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m     \u001b[1;32mfor\u001b[0m \u001b[0mchromosome\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mpopulation\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 4\u001b[1;33m         \u001b[0mlogmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mchromosome\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my_train\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      5\u001b[0m         \u001b[0mpredictions\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlogmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_test\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mchromosome\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      6\u001b[0m         \u001b[0mscores\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0maccuracy_score\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mpredictions\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_logistic.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[0;32m   1599\u001b[0m                       \u001b[0mpenalty\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mpenalty\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmax_squared_sum\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmax_squared_sum\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1600\u001b[0m                       sample_weight=sample_weight)\n\u001b[1;32m-> 1601\u001b[1;33m             for class_, warm_start_coef_ in zip(classes_, warm_start_coef))\n\u001b[0m\u001b[0;32m   1602\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1603\u001b[0m         \u001b[0mfold_coefs_\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0m_\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mn_iter_\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mzip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mfold_coefs_\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, iterable)\u001b[0m\n\u001b[0;32m   1002\u001b[0m             \u001b[1;31m# remaining jobs.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1003\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_iterating\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;32mFalse\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1004\u001b[1;33m             \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdispatch_one_batch\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1005\u001b[0m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_iterating\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_original_iterator\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1006\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36mdispatch_one_batch\u001b[1;34m(self, iterator)\u001b[0m\n\u001b[0;32m    833\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[1;32mFalse\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    834\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 835\u001b[1;33m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_dispatch\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtasks\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    836\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[1;32mTrue\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    837\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m_dispatch\u001b[1;34m(self, batch)\u001b[0m\n\u001b[0;32m    752\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_lock\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    753\u001b[0m             \u001b[0mjob_idx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_jobs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 754\u001b[1;33m             \u001b[0mjob\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mapply_async\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcallback\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mcb\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    755\u001b[0m             \u001b[1;31m# A job can complete so quickly than its callback is\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    756\u001b[0m             \u001b[1;31m# called before we get here, causing self._jobs to\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py\u001b[0m in \u001b[0;36mapply_async\u001b[1;34m(self, func, callback)\u001b[0m\n\u001b[0;32m    207\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mapply_async\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcallback\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    208\u001b[0m         \u001b[1;34m\"\"\"Schedule a func to be run\"\"\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 209\u001b[1;33m         \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mImmediateResult\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfunc\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    210\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mcallback\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    211\u001b[0m             \u001b[0mcallback\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, batch)\u001b[0m\n\u001b[0;32m    588\u001b[0m         \u001b[1;31m# Don't delay the application, to avoid keeping the input\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    589\u001b[0m         \u001b[1;31m# arguments in memory\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 590\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mresults\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mbatch\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    591\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    592\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    254\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mparallel_backend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mn_jobs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_n_jobs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    255\u001b[0m             return [func(*args, **kwargs)\n\u001b[1;32m--> 256\u001b[1;33m                     for func, args, kwargs in self.items]\n\u001b[0m\u001b[0;32m    257\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    258\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m__len__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\joblib\\parallel.py\u001b[0m in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m    254\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mparallel_backend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mn_jobs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_n_jobs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    255\u001b[0m             return [func(*args, **kwargs)\n\u001b[1;32m--> 256\u001b[1;33m                     for func, args, kwargs in self.items]\n\u001b[0m\u001b[0;32m    257\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    258\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m__len__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_logistic.py\u001b[0m in \u001b[0;36m_logistic_regression_path\u001b[1;34m(X, y, pos_class, Cs, fit_intercept, max_iter, tol, verbose, solver, coef, class_weight, dual, penalty, intercept_scaling, multi_class, random_state, check_input, max_squared_sum, sample_weight, l1_ratio)\u001b[0m\n\u001b[0;32m    934\u001b[0m                 \u001b[0mfunc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mw0\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmethod\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"L-BFGS-B\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mjac\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    935\u001b[0m                 \u001b[0margs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtarget\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1.\u001b[0m \u001b[1;33m/\u001b[0m \u001b[0mC\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0msample_weight\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 936\u001b[1;33m                 \u001b[0moptions\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m{\u001b[0m\u001b[1;34m\"iprint\"\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0miprint\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"gtol\"\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mtol\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"maxiter\"\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mmax_iter\u001b[0m\u001b[1;33m}\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    937\u001b[0m             )\n\u001b[0;32m    938\u001b[0m             n_iter_i = _check_optimize_result(\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\scipy\\optimize\\_minimize.py\u001b[0m in \u001b[0;36mminimize\u001b[1;34m(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)\u001b[0m\n\u001b[0;32m    608\u001b[0m     \u001b[1;32melif\u001b[0m \u001b[0mmeth\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m'l-bfgs-b'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    609\u001b[0m         return _minimize_lbfgsb(fun, x0, args, jac, bounds,\n\u001b[1;32m--> 610\u001b[1;33m                                 callback=callback, **options)\n\u001b[0m\u001b[0;32m    611\u001b[0m     \u001b[1;32melif\u001b[0m \u001b[0mmeth\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m'tnc'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    612\u001b[0m         return _minimize_tnc(fun, x0, args, jac, bounds, callback=callback,\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\scipy\\optimize\\lbfgsb.py\u001b[0m in \u001b[0;36m_minimize_lbfgsb\u001b[1;34m(fun, x0, args, jac, bounds, disp, maxcor, ftol, gtol, eps, maxfun, maxiter, iprint, callback, maxls, **unknown_options)\u001b[0m\n\u001b[0;32m    343\u001b[0m             \u001b[1;31m# until the completion of the current minimization iteration.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    344\u001b[0m             \u001b[1;31m# Overwrite f and g:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 345\u001b[1;33m             \u001b[0mf\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mg\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfunc_and_grad\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    346\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[0mtask_str\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstartswith\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34mb'NEW_X'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    347\u001b[0m             \u001b[1;31m# new iteration\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\scipy\\optimize\\lbfgsb.py\u001b[0m in \u001b[0;36mfunc_and_grad\u001b[1;34m(x)\u001b[0m\n\u001b[0;32m    293\u001b[0m     \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    294\u001b[0m         \u001b[1;32mdef\u001b[0m \u001b[0mfunc_and_grad\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 295\u001b[1;33m             \u001b[0mf\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfun\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    296\u001b[0m             \u001b[0mg\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mjac\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    297\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mg\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\scipy\\optimize\\optimize.py\u001b[0m in \u001b[0;36mfunction_wrapper\u001b[1;34m(*wrapper_args)\u001b[0m\n\u001b[0;32m    325\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mfunction_wrapper\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mwrapper_args\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    326\u001b[0m         \u001b[0mncalls\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m+=\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 327\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mfunction\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mwrapper_args\u001b[0m \u001b[1;33m+\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    328\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    329\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0mncalls\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfunction_wrapper\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mF:\\Anaconda\\Anaconda3\\lib\\site-packages\\scipy\\optimize\\optimize.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, x, *args)\u001b[0m\n\u001b[0;32m     62\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     63\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m__call__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mx\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 64\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnumpy\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcopy\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     65\u001b[0m         \u001b[0mfg\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfun\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     66\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mjac\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mfg\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
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
