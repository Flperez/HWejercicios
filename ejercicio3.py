import numpy as np
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--detection",required = False, help = "ruta a donde se guarda el archivo de detection.csv")
ap.add_argument("-g","--groundtruth",required = False, help = "ruta a donde se guarda el archivo de groundtruth.csv")


args = vars(ap.parse_args())

path_detection = args['detection']
path_groundtruth = args['groundtruth']


groundtruth = np.genfromtxt(fname=path_groundtruth,delimiter=',',skip_header=1,missing_values='-')
detection = np.genfromtxt(fname=path_detection,delimiter=',',skip_header=1,missing_values='-')


if __name__ == "__main__":

    result = abs(groundtruth-detection)



    rango_areas = 0, 0,50,100,150,200,250
    num = np.zeros(7)
    rango_complexity = 0,1,2,3,4


    for i in range(0,len(result)):

        if np.isnan(result[i][1])==True:
            k=0
        else:
            for j in range(1,len(rango_areas)-1):

                if result[i][1]>rango_areas[j] and result[i][1]<rango_areas[j+1]:
                    k=j
                    break
                else:
                    k=6

        print("R[", i, "][1]: ", result[i][1])
        print(k)
        num[k]=num[k]+1



    print(num)
    num = (1/len(result))*100*num
    print(num)

    #Figure area2d
    N = 7
    ind = np.arange(N)

    fig, ax = plt.subplots()
    rects = ax.bar(ind, num, 0.5, color='r')


    ax.set_xticklabels(('[0]','[error]', '[0-50]', '[50-100]','[100-150]','[150-200]','[200-250]','>250'))
    plt.xlabel('Squared feet error' )
    plt.ylabel('Percentage of blueprints' )
    plt.title('Area2D' )
    plt.show()
