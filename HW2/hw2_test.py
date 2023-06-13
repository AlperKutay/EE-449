#Importing necessary libraries
import cv2
from random import randint
from random import random
from random import uniform
import random as rnd
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import json
#Initialize variables with an input image
input_path='painting.png'
im = cv2.imread(input_path)
width,height,_ = im.shape
WIDTH,HEIGHT,_ = im.shape
# Reference to :https://note.nkmk.me/en/python-opencv-pillow-image-size/
print("Widht is ", width)
print("Height is ",height)

num_inds = [5, 10, 20, 40, 60]
num_genes = [15, 30, 50, 80, 120]
tm_size = [2, 5, 8, 16]
frac_elites = [0.04, 0.2, 0.35]
frac_parents = [0.15, 0.3, 0.6, 0.75]
mutation_prob = [0.1, 0.2, 0.4, 0.75]
mutation_type = ["guided", "unguided"]
num_generation = 10000
dictionary = {"fitness" : None,}


class HyperParameters:
  def __init__(self, num_inds, num_genes, tm_size, num_elites, num_parents, mutation_prob, mutation_type):#collect parameters in a class
    self.num_inds = num_inds
    self.num_genes = num_genes
    self.tm_size = tm_size
    self.frac_elites = num_elites
    self.frac_parents = num_parents
    self.mutation_prob = mutation_prob
    self.mutation_type = mutation_type
#Gene Class 
#----------------------------------------------------------------------------------------------------------
class Gene:
    def __init__(self, index=-1, x=0, y=0, rad=1, R=0, G=0, B=0, A=0):
        self.index = index
        self.x = x
        self.y = y
        self.rad = rad
        self.R = R
        self.G = G
        self.B = B
        self.A = A

    def create_gene(self, index):#Create genes
        self.R = rnd.randrange(256)
        self.G = rnd.randrange(256)
        self.B = rnd.randrange(256)
        self.index = index
        self.A = rnd.random()
        temp_x = rnd.randrange(int(1.5 * WIDTH))
        temp_y = rnd.randrange(int(1.5 * HEIGHT))
        temp_rad = rnd.randrange(int(max(WIDTH, HEIGHT) / 2))
        while not self.isIntersects(temp_x, temp_y, temp_rad):
            temp_x = rnd.randrange(int(1.5 * WIDTH))
            temp_y = rnd.randrange(int(1.5 * HEIGHT))
            temp_rad = rnd.randrange(int(max(WIDTH, HEIGHT) / 2))
        self.x = temp_x
        self.y = temp_y
        self.rad = temp_rad

    def isIntersects(self, x, y, r):#it checks whether genes are valid or not 
        dist_x = abs(x - WIDTH / 2)
        dist_y = abs(y - HEIGHT / 2)

        if dist_x > (WIDTH / 2 + r): return False
        if dist_y > (HEIGHT / 2 + r): return False

        if dist_x <= (WIDTH / 2): return True
        if dist_y <= (HEIGHT / 2): return True

        cornerDistance_sq = (dist_x - WIDTH / 2) ** 2 + (dist_y - HEIGHT / 2) ** 2
        return cornerDistance_sq <= (r ** 2)
    
    def guided_Mutation(self):#it provides guided_mutation according to information given in manual
        #it actually does same things with create_gene() but there is some limitations which is written
        temp_x = rnd.randrange(max(0, int(self.x-WIDTH/4)), int(self.x+WIDTH/4)+1)
        temp_y = rnd.randrange(max(0, int(self.y-HEIGHT/4)), int(self.y+HEIGHT/4)+1)
        temp_rad = rnd.randrange(max(0, self.rad-10), self.rad+11)
        while not self.isIntersects(temp_x, temp_y, temp_rad):
            temp_x = rnd.randrange(max(0, int(self.x-WIDTH/4)), int(self.x+WIDTH/4)+1)
            temp_y = rnd.randrange(max(0, int(self.y-HEIGHT/4)), int(self.y+HEIGHT/4)+1)
            temp_rad = rnd.randrange(max(0, self.rad-10), self.rad+11)
        self.x = temp_x
        self.y = temp_y
        self.rad = temp_rad
        self.R =  rnd.randrange(max(0, self.R-64), min(self.R+65, 255))
        self.G =  rnd.randrange(max(0, self.G-64), min(self.G+65, 255))
        self.B =  rnd.randrange(max(0, self.B-64), min(self.B+65, 255))
        rnd_a = rnd.random()/2.0 - 0.25
        self.A =  max(0, min(1.0, rnd_a + self.A))
    
    def mutate(self,ref,index):
        if ref == "G":
            self.guided_Mutation()
        else:
            self.create_gene(index)
    def printGene(self):
        print(f"Gene - {self.index}: x:{self.x}, y:{self.y}, RAD:{self.rad}, R:{self.R}, G:{self.G}, B:{self.B},A:{self.A}")
            
#------------------------------------------------------------------------------------------------------
#Individual Class
class indv:
    def __init__(self,ID=-1,num_genes=50):
        #Constructor Method
        self.ID=ID
        self.num_genes=num_genes
        #Create empty list for choromosomes
        self.chromosome = list()
        for i in range(1,num_genes+1):
            bi=Gene()#create temp gene for chromosome
            bi.create_gene(i)
            self.chromosome.append(bi)
            
    def evaulation(self):
        self.chromosome.sort(key=lambda x: x.rad, reverse=True)
        image = np.full((width, height, 3),255, dtype = np.uint8)
        for gene in self.chromosome:
            overlay=deepcopy(image)
            cv2.circle(overlay, (gene.x, gene.y), gene.rad, (gene.B, gene.G, gene.R), -1)
            image = cv2.addWeighted(overlay, gene.A, image, (1-gene.A), 0.0,image)
        diff=np.subtract(np.array(im, dtype=np.int64), np.array(image, dtype=np.int64))
        self.fitness = np.sum(-1*np.power(diff, 2))
            
    def created_image(self):#This method will be called when a image is reqiured
        self.chromosome.sort(key=lambda x: x.rad, reverse=True)
        image = np.full((width, height, 3),255, dtype = np.uint8)
        for gene in self.chromosome:
            overlay=deepcopy(image)
            cv2.circle(overlay, (gene.x, gene.y), gene.rad, (gene.B, gene.G, gene.R), -1)
            image = cv2.addWeighted(overlay, gene.A, image, (1-gene.A), 0.0,image)
        return image
    
    def calculate_fitness(self):
        #Reference to:
        #https://www.techiedelight.com/sort-list-of-objects-python/#:~:text=A%20simple%20solution%20is%20to,only%20arguments%3A%20key%20and%20reverse.
        self.chromosome.sort(key=lambda x: x.rad, reverse=True)
        
        #Reference to:
        #https://www.geeksforgeeks.org/create-a-white-image-using-numpy-in-python/
        image = np.full((width, height, 3),255, dtype = np.uint8)
        for i in self.chromosome:
            #overlay <- image 
            #Avoid shallow copy issues, assign it with deep copy
            #Reference to :
            #https://docs.python.org/3/library/copy.html
            overlay=deepcopy(image)
            
            # Draw the circle on overlay.
            # Reference to:
            # https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
            # Center coordinates
            center_coordinates = (i.x, i.y)
            # Radius of circle
            radius = i.rad
            # color in BGR
            color = (i.B, i.G, i.R)
            # Line thickness of -1 px
            thickness = -1
            cv2.circle(overlay, center_coordinates, radius, color, thickness)
            
            # image <- overlay x alpha + image x (1-alpha)
            # Reference to:
            # https://www.educba.com/opencv-addweighted/
            image = cv2.addWeighted(overlay, i.A, image, (1-i.A), 0.0,image)
        
        #Calculating fitness of INDV
        diff=np.subtract(np.array(im, dtype=np.int32), np.array(image, dtype=np.int32))
        self.fitness = np.sum(-1*np.power(diff, 2))
        if self.fitness >0:
            print("ERROR----------------------------------------------------------")
            #Undesired situtation 
            #Code should not be entered this region

    def takeImage(self):#This method will be called when a image is reqiured
        #Reference to:
        #https://www.techiedelight.com/sort-list-of-objects-python/#:~:text=A%20simple%20solution%20is%20to,only%20arguments%3A%20key%20and%20reverse.
        self.chromosome.sort(key=lambda x: x.rad, reverse=True)
        
        #Reference to:
        #https://www.geeksforgeeks.org/create-a-white-image-using-numpy-in-python/
        image = np.full((width, height, 3),255, dtype = np.uint8)
        for i in self.chromosome:
            #overlay <- image 
            #Avoid shallow copy issues, assign it with deep copy
            #Reference to :
            #https://docs.python.org/3/library/copy.html
            overlay=deepcopy(image)
            
            # Draw the circle on overlay.
            # Reference to:
            # https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
            # Center coordinates
            center_coordinates = (i.x, i.y)
            # Radius of circle
            radius = i.rad
            # color in BGR
            color = (i.B, i.G, i.R)
            # Line thickness of -1 px
            thickness = -1
            cv2.circle(overlay, center_coordinates, radius, color, thickness)
            
            # image <- overlay x alpha + image x (1-alpha)
            # Reference to:
            # https://www.educba.com/opencv-addweighted/
            image = cv2.addWeighted(overlay, i.A, image, (1-i.A), 0.0,image)
        return image
    
    def mutation(self,prob,guide):
        while(random()<prob):#Do it with a given probablity
            #mutate random gene in the chromosome
            genMutated= randint(0,self.num_genes-1)
            if(guide=="guided"):#guided
                self.chromosome[genMutated].mutate("G",self.chromosome[genMutated].index)

            else:#unguided
                self.chromosome[genMutated].mutate("U",self.chromosome[genMutated].index)
    
    def printIndividual(self):
        #print("Individual -",self.index)
        print("Fitness: ", self.fitness)
        #print("Chrosome:")
        #for gene in self.chro:
        #    gene.printGene()
                                
#-------------------------------------------------------------------------------------------------------
#Population Class
class popu:
    def __init__(self,hyp_parameters,name,num_of_generations):
        #Constructor method for population
        self.mutation_prob=hyp_parameters.mutation_prob
        self.mutation_guide=hyp_parameters.mutation_type
        self.num_iteration=num_of_generations
        self.frac_elites=hyp_parameters.frac_elites
        self.frac_parents=hyp_parameters.frac_parents
        self.name=name
        self.num_idv=hyp_parameters.num_inds
        self.num_genes=hyp_parameters.num_genes
        self.tm_size=hyp_parameters.tm_size
        #Clear individual lists
        self.indvs = list()
        for i in range(0,self.num_idv):
            bi=indv(i,self.num_genes)#Temp Indv to add to population
            self.indvs.append(bi)
            #Create individual
    def eval_population(self):
        for indv in self.indvs:#calculate each one of the indvs in population
            indv.calculate_fitness()
            
    def printPopulation(self):
      for ind in self.indvs:
        ind.printIndividual()
        
    def evaluation(self):#This method is our main method to train, it will train our model and record all necessary information
        fitness_values = []#This will hold all fitness values corresponds to population
        self.parents=list()#Parents will hold the individuals who will have children
        self.children=list()#Children will hold the individuals who is created with crossover
        self.elites=list()#Elites will hold the individuals who can go to next generation directly
        self.others=list()
        for i in range(self.num_iteration+1):
            self.eval_population()
            #Calculate number of elites and parents
            self.num_elites=int(self.frac_elites * self.num_idv)
            self.num_parent=int(self.frac_parents * self.num_idv)
            #if it is even number add 1
            if self.num_parent % 2 == 1:
                self.num_parent = self.num_parent + 1
            self.indvs.sort(key=lambda x: x.fitness, reverse=True)
            self.elites.clear()
            for z in range(self.num_elites):
                self.elites.append(deepcopy(self.indvs[z]))
            for _ in range(self.num_parent):
                self.parents.append(self.tournament())
            fitness_values.append(float(self.elites[0].fitness))
            for _ in range(self.num_elites):
                self.indvs.pop(0)
            self.children.clear()
            #Call the crossover method
            self.crossover()
            #Crossover Finishes --------------
           
            #Mutation Starts
            self.others.clear()
            self.mutation()
            #Mutation finishes ---------------
            #Print informative message
            if i % 10 == 9:
                print(f"Loading %{(i+1)/100},{fitness_values[i]}")
            if i % 1000 == 999:#pring png in each 1000 generations
                name = self.name+'iteration_'+str(i+1)+'.png'
                print("Name : "+name)
                cv2.imwrite(name,self.indvs[0].takeImage())
        dictonary ={
            "fitness" : fitness_values
        }
        with open(self.name+"fitnes_values.json","w") as outfile:
            json.dump(dictonary,outfile)
        del fitness_values
    
    def eval_2_1(self):#This method is our main method to train, it will train our model and record all necessary information
        fitness_values = []#This will hold all fitness values corresponds to population
        self.parents=list()#Parents will hold the individuals who will have children
        self.children=list()#Children will hold the individuals who is created with crossover
        self.elites=list()#Elites will hold the individuals who can go to next generation directly
        self.others=list()
        for i in range(self.num_iteration+1):
            self.eval_population()
            #Calculate number of elites and parents
            self.num_elites=int(self.frac_elites * self.num_idv)
            self.num_parent=int(self.frac_parents * self.num_idv)
            #if it is even number add 1
            if self.num_parent % 2 == 1:
                self.num_parent = self.num_parent + 1
            self.indvs.sort(key=lambda x: x.fitness, reverse=True)
            self.elites.clear()
            for z in range(self.num_elites):
                self.elites.append(deepcopy(self.indvs[z]))
            for _ in range(self.num_parent):
                self.parents.append(self.tournament())
            fitness_values.append(float(self.elites[0].fitness))
            for _ in range(self.num_elites):
                self.indvs.pop(0)
            self.children.clear()
            #Call the crossover method
            self.crossover_suggestion()
            #Crossover Finishes --------------
           
            #Mutation Starts
            self.others.clear()
            self.mutation()
            #Mutation finishes ---------------
            #Print informative message
            if i % 10 == 9:
                print(f"Loading %{(i+1)/100},{fitness_values[i]}")
            if i % 1000 == 999:#pring png in each 1000 generations
                name = self.name+'iteration_'+str(i+1)+'.png'
                print("Name : "+name)
                cv2.imwrite(name,self.indvs[0].takeImage())
        dictonary ={
            "fitness" : fitness_values
        }
        with open(self.name+"fitnes_values.json","w") as outfile:
            json.dump(dictonary,outfile)
        del fitness_values
        
    def eval_2_2(self):#This method is our main method to train, it will train our model and record all necessary information
        fitness_values = []#This will hold all fitness values corresponds to population
        self.parents=list()#Parents will hold the individuals who will have children
        self.children=list()#Children will hold the individuals who is created with crossover
        self.elites=list()#Elites will hold the individuals who can go to next generation directly
        self.others=list()
        for i in range(self.num_iteration+1):
            self.eval_population()
            #Calculate number of elites and parents
            self.num_elites=int(self.frac_elites * self.num_idv)
            self.num_parent=int(self.frac_parents * self.num_idv)
            #if it is even number add 1
            if self.num_parent % 2 == 1:
                self.num_parent = self.num_parent + 1
            self.indvs.sort(key=lambda x: x.fitness, reverse=True)
            self.elites.clear()
            for z in range(self.num_elites):
                self.elites.append(deepcopy(self.indvs[z]))
            for _ in range(self.num_parent):
                self.parents.append(self.tournament())
            fitness_values.append(float(self.elites[0].fitness))
            for _ in range(self.num_elites):
                self.indvs.pop(0)
            self.children.clear()
            #Call the crossover method
            self.crossover()
            #Crossover Finishes --------------
           
            #Mutation Starts
            self.others.clear()
            self.mutation_suggestion()
            #Mutation finishes ---------------
            #Print informative message
            if i % 10 == 9:
                print(f"Loading %{(i+1)/100},{fitness_values[i]}")
            if i % 1000 == 999:#pring png in each 1000 generations
                name = self.name+'iteration_'+str(i+1)+'.png'
                print("Name : "+name)
                cv2.imwrite(name,self.indvs[0].takeImage())
        dictonary ={
            "fitness" : fitness_values
        }
        with open(self.name+"fitnes_values.json","w") as outfile:
            json.dump(dictonary,outfile)
        del fitness_values       

    def eval_2_3(self):#This method is our main method to train, it will train our model and record all necessary information
        fitness_values = []#This will hold all fitness values corresponds to population
        self.parents=list()#Parents will hold the individuals who will have children
        self.children=list()#Children will hold the individuals who is created with crossover
        self.elites=list()#Elites will hold the individuals who can go to next generation directly
        self.others=list()
        for i in range(self.num_iteration+1):
            self.adjust_mut_parameters(i)
            self.eval_population()
            #Calculate number of elites and parents
            self.num_elites=int(self.frac_elites * self.num_idv)
            self.num_parent=int(self.frac_parents * self.num_idv)
            #if it is even number add 1
            if self.num_parent % 2 == 1:
                self.num_parent = self.num_parent + 1
            self.indvs.sort(key=lambda x: x.fitness, reverse=True)
            self.elites.clear()
            for z in range(self.num_elites):
                self.elites.append(deepcopy(self.indvs[z]))
            for _ in range(self.num_parent):
                self.parents.append(self.tournament())
            fitness_values.append(float(self.elites[0].fitness))
            for _ in range(self.num_elites):
                self.indvs.pop(0)
            self.children.clear()
            #Call the crossover method
            self.crossover()
            #Crossover Finishes --------------
           
            #Mutation Starts
            self.others.clear()
            self.mutation_suggestion()
            #Mutation finishes ---------------
            #Print informative message
            if i % 10 == 9:
                print(f"Loading %{(i+1)/100},{fitness_values[i]}")
            if i % 1000 == 999:#pring png in each 1000 generations
                name = self.name+'iteration_'+str(i+1)+'.png'
                print("Name : "+name)
                cv2.imwrite(name,self.indvs[0].takeImage())
        dictonary ={
            "fitness" : fitness_values
        }
        with open(self.name+"fitnes_values.json","w") as outfile:
            json.dump(dictonary,outfile)
        del fitness_values        
    #Mutation method is to mutate population
    def mutation(self):
        #Create indvs list for who are applied to mutation
        #Children and other individuals(we excluded elites already )
        mutationTeam=self.children + self.indvs
        for i_indv in mutationTeam:
            i_indv.mutation(self.mutation_prob,self.mutation_guide)
        self.indvs=deepcopy( self.children +self.elites + self.indvs)
        
    def mutation_suggestion(self):
        #Create indvs list for who are applied to mutation
        #Children and other individuals(we excluded elites already )
        
        mutationTeam=self.children + self.indvs
        for i_indv in mutationTeam:
            i_indv.mutation(self.mutation_prob,self.mutation_guide)
        self.indvs.sort(key=lambda  item:item.fitness , reverse=True)
        new1 = indv(self.indvs[-1].ID,self.num_genes)
        new2 = indv(self.indvs[-2].ID,self.num_genes)
        new1.calculate_fitness()
        new2.calculate_fitness()
        self.indvs.pop(len(self.indvs)-1)
        self.indvs.pop(len(self.indvs)-1)
        self.indvs.append(new1)
        self.indvs.append(new2)
        self.indvs=deepcopy( self.children +self.elites + self.indvs)
        
    #Crossover method
    #This will update children list with newly created 
        #individuals crossovering parents
    def crossover(self):
        for _ in range(0,self.num_parent,2):#Iterate the amount of parents divided by two
                                            #Since each children has two parents
            father=self.parents.pop(randint(0,len(self.parents)-1))#Randomly assign father
            mother=self.parents.pop(randint(0,len(self.parents)-1))#Randomly assign mother
            childrenA=indv(father.ID,self.num_genes)#Create new children
            childrenB=indv(mother.ID,self.num_genes)#Create new children
            #For each gen, randomize a number between 0 and 1
            # if it is smaller than 0.5
            #father will give the gene to children
            #if not
            #mother will give the gene to children
            for i in range(0,self.num_genes):
                res=uniform(0,1)
                if res<0.5:
                    childrenA.chromosome[i]=father.chromosome[i]
                    childrenB.chromosome[i]=mother.chromosome[i]
                else:
                    childrenA.chromosome[i]=mother.chromosome[i]
                    childrenB.chromosome[i]=father.chromosome[i]
            #Update the children list to 
            self.children.append(childrenA)
            self.children.append(childrenB)
    #Tourname method will give us a winner
    def crossover_suggestion(self):
        for _ in range(0,self.num_parent,2):#Iterate the amount of parents divided by two
                                            #Since each children has two parents
            family = []
            father=self.parents.pop(randint(0,len(self.parents)-1))#Randomly assign father
            mother=self.parents.pop(randint(0,len(self.parents)-1))#Randomly assign mother
            childrenA=indv(father.ID,self.num_genes)#Create new children
            childrenB=indv(mother.ID,self.num_genes)#Create new children
            #For each gen, randomize a number between 0 and 1
            # if it is smaller than 0.5
            #father will give the gene to children
            #if not
            #mother will give the gene to children
            for i in range(0,self.num_genes):
                res=uniform(0,1)
                if res<0.5:
                    childrenA.chromosome[i]=father.chromosome[i]
                    childrenB.chromosome[i]=mother.chromosome[i]
                else:
                    childrenA.chromosome[i]=mother.chromosome[i]
                    childrenB.chromosome[i]=father.chromosome[i]
            #Update the children list to 
            childrenA.calculate_fitness()
            childrenB.calculate_fitness()
            family.append(deepcopy(childrenA))
            family.append(deepcopy(childrenB))
            family.append(deepcopy(father))
            family.append(deepcopy(mother))
            family= sorted(family,key=lambda item : item.fitness,reverse=True)
            self.children.append(family[0])
            self.children.append(family[1])
            family.clear()
            
    def tournament(self):
        #Randomly choose one of them
        #Assign it as temporary winner
        bestInd=randint(0,len(self.indvs)-1)
        bestFitness=self.indvs[bestInd].fitness
        
        #Since we initialize with a random assignment
        #One of the warrior is decided
        #Therefore, we should iterate tm_size - 1 times 
        for _ in range(self.tm_size-1):
            #Randomly choose one of them as a warrior
            currentInd=randint(0,len(self.indvs)-1)
            #Take the warrior's fitness value
            currentFitness=self.indvs[currentInd].fitness
            #Compare it with the best one
            if(currentFitness > bestFitness):
                #If the iterated warrior wins
                #Label him as best
                bestFitness=currentFitness
                bestInd=currentInd
        #end of for loop
        #Winner is decided
        #Temporary indv to be added to parent since it is the winner.
        temp=self.indvs[bestInd]
        self.indvs.pop(bestInd)#Delete the winner from current generation
        return temp#Return it so that parents will be updated correctly
    def adjust_mut_parameters(self,iteration):
        if iteration < 1000:
            self.mutation_type = 'unguided'
            self.mutation_prob = 0.8
        elif 6000> iteration >5000:
            self.mutation_type = 'guided'
            self.mutation_prob = 0.5

"""
#suggestion 1 : choose best of two from family  crossover changed 
params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="discussion/evol_2_1/",num_of_generations=num_generation)
pop.eval_2_1()  


#suggestion 2 : get rid of the others(not elits and children) and create new Indiviuals mutation changed
params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop2 = popu(hyp_parameters=params,name="discussion/evol_2_2/",num_of_generations=num_generation)
pop2.eval_2_2() 
#?
"""
#evolution2_3 changable mut type and prob 
params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop3 = popu(hyp_parameters=params,name="discussion/evol_2_3/",num_of_generations=num_generation)
pop3.eval_2_3() 
#?

"""
#default    
params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="num_inds/"+str(num_inds[2])+"/",num_of_generations=num_generation)
pop.evaluation()
        
 
#inds
params = HyperParameters(num_inds[0], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="num_inds/"+str(num_inds[0])+"/",num_of_generations=num_generation)
pop.evaluation()

params = HyperParameters(num_inds[1], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="num_inds/"+str(num_inds[1])+"/",num_of_generations=num_generation)
pop.evaluation()

params = HyperParameters(num_inds[3], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="num_inds/"+str(num_inds[3])+"/",num_of_generations=num_generation)
pop.evaluation()

params = HyperParameters(num_inds[4], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="num_inds/"+str(num_inds[4])+"/",num_of_generations=num_generation)
pop.evaluation()



#genes

params = HyperParameters(num_inds[2], num_genes[0], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="num_genes/"+str(num_genes[0])+"/",num_of_generations=num_generation)
pop.evaluation()   

params = HyperParameters(num_inds[2], num_genes[1], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="num_genes/"+str(num_genes[1])+"/",num_of_generations=num_generation)
pop.evaluation()   

params = HyperParameters(num_inds[2], num_genes[3], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="num_genes/"+str(num_genes[3])+"/",num_of_generations=num_generation)
pop.evaluation()   

params = HyperParameters(num_inds[2], num_genes[4], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="num_genes/"+str(num_genes[4])+"/",num_of_generations=num_generation)
pop.evaluation()   

 


#tm_size
params = HyperParameters(num_inds[2], num_genes[2], tm_size[0], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="tm_size/"+str(tm_size[0])+"/",num_of_generations=num_generation)
pop.evaluation()   

params = HyperParameters(num_inds[2], num_genes[2], tm_size[2], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="tm_size/"+str(tm_size[2])+"/",num_of_generations=num_generation)
pop.evaluation()   

params = HyperParameters(num_inds[2], num_genes[2], tm_size[3], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="tm_size/"+str(tm_size[3])+"/",num_of_generations=num_generation)
pop.evaluation()   

"""
"""
#elits
params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[0], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="elites/"+str(int(frac_elites[0]*100))+"/",num_of_generations=num_generation)
pop.evaluation()  


params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[2], frac_parents[2], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="elites/"+str(int(frac_elites[2]*100))+"/",num_of_generations=num_generation)
pop.evaluation()   


#parents
params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[1], frac_parents[0], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="parents/"+str(int(frac_parents[0]*100))+"/",num_of_generations=num_generation)
pop.evaluation()   

params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[1], frac_parents[1], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="parents/"+str(int(frac_parents[1]*100))+"/",num_of_generations=num_generation)
pop.evaluation()   


params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[1], frac_parents[3], mutation_prob[1], mutation_type[0])
pop = popu(hyp_parameters=params,name="parents/"+str(int(frac_parents[3]*100))+"/",num_of_generations=num_generation)
pop.evaluation()   


#mut_prob
params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[0], mutation_type[0])
pop = popu(hyp_parameters=params,name="mut_prob/"+str(int(mutation_prob[0]*100))+"/",num_of_generations=num_generation)
pop.evaluation()   



params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[2], mutation_type[0])
pop = popu(hyp_parameters=params,name="mut_prob/"+str(int(mutation_prob[2]*100))+"/",num_of_generations=num_generation)
pop.evaluation()   



params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], 0.95, mutation_type[0])
pop = popu(hyp_parameters=params,name="mut_prob/"+str(int(mutation_prob[3]*100))+"/",num_of_generations=num_generation)
pop.evaluation()   

#type

params = HyperParameters(num_inds[2], num_genes[2], tm_size[1], frac_elites[1], frac_parents[2], mutation_prob[1], mutation_type[1])
pop = popu(hyp_parameters=params,name="mut_type/"+str(mutation_type[1])+"/",num_of_generations=num_generation)
pop.evaluation()    

"""