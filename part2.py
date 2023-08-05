## Thomas Pham Net ID: TTP190005 
## Christopher Chan Net ID: CCC180002

## Libraries
from pathlib import Path
from typing import Union
import re
from typing import Tuple
import random
import math
from collections import defaultdict


## Fetch the dataset and remove symbols
def preProcessData(
  filePath: Union[str, Path],
) -> list:
  
  myFile = open(filePath, "r")
  rawTweets = myFile.readlines()
  cleanedTweetsList = []

  for rawTweet in rawTweets:

    rawTweet = rawTweet.replace('@', '')                #just @symbol
    rawTweet = re.sub(r'^\d+\|.*?\|', '', rawTweet)     #tweet ID and timestamp
    rawTweet = re.sub(r'@\w+', '', rawTweet)            #@tag
    rawTweet = re.sub(r'#', '', rawTweet)               #symbol
    rawTweet = re.sub(r'http\S+', '', rawTweet)         #http link

    rawTweet = rawTweet.replace('"', '')                #double quote
    rawTweet = rawTweet.replace("'", '')                #single quote
    rawTweet = rawTweet.replace(',', '')                #comma
    rawTweet = rawTweet.replace('?', '')                #question mark
    rawTweet = rawTweet.replace('!', '')                #exclamation mark
    rawTweet = rawTweet.replace('|', '')                #pipe
    rawTweet = rawTweet.replace('(', '')                #parenthesis open
    rawTweet = rawTweet.replace(')', '')                #parenthesis close
    rawTweet = rawTweet.replace('-', ' ')               #hyphen
    rawTweet = rawTweet.replace('\\', ' ')              #back slash

    rawTweet = rawTweet.replace('w/', 'with ')          #abbreviation s
    rawTweet = rawTweet.replace('w/o', 'without ')
    rawTweet = rawTweet.replace('w/out', 'without')
    rawTweet = rawTweet.replace('h/t', 'hat tip ')
    rawTweet = rawTweet.replace('/', ' ')               #slash

    rawTweet = rawTweet.replace('\u2014', ' ')          #em dash
    rawTweet = rawTweet.replace('\u2013', ' ')          #en dash
    rawTweet = rawTweet.replace('\u2018', '')           #single quote open
    rawTweet = rawTweet.replace('\u2019', '')           #single quote close
    rawTweet = rawTweet.replace('\u201C', ' ')          #double quote open
    rawTweet = rawTweet.replace('\u201D', ' ')          #double quote close
    rawTweet = rawTweet.replace('\u2026', ' ')          #horizontal ellipsis
    rawTweet = rawTweet.replace(":", '')                #colon
    rawTweet = rawTweet.replace(";", '')                #semicolon
    rawTweet = rawTweet.lower()                         

    # Replace consecutive spaces with single space
    rawTweet = re.sub(r'\s+', ' ', rawTweet).strip()

    cleanedTweetsList.append(rawTweet)
    
  myFile.close()
  return cleanedTweetsList
  


## If the centroid has not changed, then Kmeans has converged
def isConverged(
  prevCentroids: list,
  currCentroids: list
) -> bool:
  if prevCentroids == currCentroids:
    return True
  
  return False
  
## Assign the tweets to the closest centroid and assign them to cluster
def assignClusters(
  tweets: list,
  centroids: list,
) -> list:
  clusters = defaultdict(list)

  for tweetIdx, tweet in enumerate(tweets):
    minDistance = float('inf')
    clusterIdx = -1

    for centroidIdx, centroid in enumerate(centroids):
      distance = getDistance(centroid, tweet)

      if centroid == tweet:
        clusterIdx = centroidIdx
        minDistance = 0
      
      if distance < minDistance:
        clusterIdx = centroidIdx
        minDistance = distance
    
    if minDistance == 1:
      clusterIdx = random.randint(0, len(centroids) - 1)
    
    clusters[clusterIdx].append([tweet, minDistance])

  return clusters


## After getting the cluster assignments for the tweeets, update the new centroids
def updateCentroids(
  clusters: list
) -> list:
  centroids = []

  for c in range(len(clusters)):
    minDist = math.inf
    centroidIndex = -1

    minDisDp = []

    for tweet1 in range(len(clusters[c])):
      minDisDp.append([])
      distSum = 0
      for tweet2 in range(len(clusters[c])):
        if tweet1 != tweet2:
          if tweet2 < tweet1:
            dis = minDisDp[tweet2][tweet1]
          else:
            dis = getDistance(clusters[c][tweet1][0], clusters[c][tweet2][0])

          minDisDp[tweet1].append(dis)
          distSum += dis
        else:
          minDisDp[tweet1].append(0)

      if distSum < minDist:
        minDist = distSum
        centroidIndex = tweet1

    centroids.append(clusters[c][centroidIndex][0])


  return centroids


## Calculate the Jaccard Distance
def getDistance(
  rawTweet1: str, 
  rawTweet2: str,
  ):
  tweet1Words = rawTweet1.split(' ')
  tweet2Words = rawTweet2.split(' ')
  intersection = []
  for word in tweet1Words:
    if word in tweet2Words:
      intersection.append(word)

  union = tweet1Words + [word for word in tweet2Words if word not in tweet1Words]


  jaccardDist = 1 - (len(intersection) / len(union))
  return jaccardDist

## Compute the SSE
def computeSSE(
  clusters: list
  ) -> float:

  sse = 0
  for cluster in range(len(clusters)):
    for tweet in range(len(clusters[cluster])):
      sse = sse + (clusters[cluster][tweet][1] * clusters[cluster][tweet][1])

  return sse

## Initialize centroids randomly
def initializeCentroids(
  tweets: list,
  k: int
) -> list:
  centroids = []
  while len(centroids) < k:
    getRandomTweetIndex = random.randint(0, len(tweets) -1)
    if tweets[getRandomTweetIndex] in centroids:
      continue
    else:
      centroids.append(tweets[getRandomTweetIndex])

  return centroids

## KMeans implementation
def kMeans(
  tweets: list,
  k: int,
  maxIterations: int
) -> Tuple[list, int]:
  centroids = initializeCentroids(tweets, k)
  
  iterationCount = 0
  prevCentroids = []

  while(isConverged(prevCentroids, centroids) == False and (iterationCount < maxIterations)):
    
    clusters = assignClusters(tweets, centroids)

    prevCentroids = centroids
    centroids = updateCentroids(clusters)
    iterationCount += 1


  if (iterationCount == maxIterations):
    print("Max Iterations Reached, Not Converged")
  else:
    print("Converged in " + str(iterationCount) + " iterations")
  
  for cluster in range(0, len(clusters)):
    print(str(cluster+1)+": " + str(len(clusters[cluster])) + " tweets")

  sse = computeSSE(clusters)
  print("SSE: " + str(sse))
  return clusters, sse


#### HYPERPARAMETERS ####
##K Values
K_1 = 2
K_2 = 4
K_3 = 6
K_4 = 8
K_5 = 10

## Iterations
MAX_ITER = 10

## CHANGE THIS TO YOUR FILEPATH ##
FILE_PATH = r"/Users/pham/Desktop/School/Summer 2023/CS 6375/Assignment3/Health-Tweets/nytimeshealth.txt"

def main():
  tweets = preProcessData(FILE_PATH)

  kValues = []
  kValues.append(K_1)
  kValues.append(K_2)
  kValues.append(K_3)
  kValues.append(K_4)
  kValues.append(K_5)

  for k in kValues:
    print("K value: " + str(k))
    kMeans(tweets, k, MAX_ITER)
    print('\n')
  
  
  #kMeans(tweets, 5, 10)
main()