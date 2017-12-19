import numpy as np
from matplotlib import pyplot as plt
#from scipy import stats

explored=[]

class Tree(object):
    def __init__(self,location):
        self.root=False
        self.level=0
        self.children=[]
        self.location=location
        self.parent=None
        self.end=False

    def addchild(self):
        plocation=self.location
        clocation=[]
        if [plocation[0]-1,plocation[1]] not in explored:
            explored.append([plocation[0]-1,plocation[1]])
            clocation.append([plocation[0]-1,plocation[1]])  #append down
        if [plocation[0],plocation[1]-1] not in explored:
            explored.append([plocation[0],plocation[1]-1])
            clocation.append([plocation[0],plocation[1]-1])  #append left
        if [plocation[0],plocation[1]+1] not in explored:
            explored.append([plocation[0],plocation[1]+1])
            clocation.append([plocation[0],plocation[1]+1])  #append right
        if [plocation[0]+1,plocation[1]] not in explored:
            explored.append([plocation[0]+1,plocation[1]])
            clocation.append([plocation[0]+1,plocation[1]])  #append up
        if len(clocation)==0:
            return -1
        for x in clocation:
            n=x[0]
            m=x[1]
            '''pn=plocation[0]
            pm=plocation[1]
            if layout_map[n][m]==layout_map[pn][pm] and self.nogate([n,m]):'''
            if n<len(layout_map) and m<len(layout_map[0]) and layout_map[n][m]!=1:
                child=Tree(x)
                child.level=self.level+1
                child.parent=self
                self.children.append(child)
        return 1


    def nogate(self,location):
        for x in gate:
            if x[0]==location[0] and x[1]==location[1]:
                return False
        return True

    def traverse(self):
        if self.end==False:
            if len(self.children)==0:
                return self
            else:
                for child in self.children:
                    if child.end==False:
                        if len(child.children)==0:
                            return child
                return None
        else:
            return None

    def traverse_gdchild(self):
        for child in self.children:
            if child.end==False:
                for gdch in child.children:
                    if len(gdch.children)==0:
                        return child





layout_map=[  #201->living room, #202->bedroom left, #203->bedroom right, #204->kitchen #205->toilet
    [1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,1],  #0
    [1,	5,	202,202,202,202,202,202,13,	202,7,	202,202,202,202,202,1,	5,	203,203,203,203,203,13,	203,7,	203,203,203,203,203,1], #1
    [1,	202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,1,	203,203,203,203,203,203,203,203,203,203,203,203,203,203,1], #2
    [1,	21,	21,	21,	21,	202,202,202,202,202,202,202,202,202,202,202,1,	22,	22,	22,	22,	203,203,203,203,203,203,203,203,203,203,1], #3
    [1,	202,202,202,202,202,202,202,202,202,202,202,202,8,	202,202,1,	203,203,203,203,203,203,203,203,203,203,203,8,	203,203,1], #4
    [1,	202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,1,	203,203,203,203,203,203,203,203,203,203,203,203,203,203,1], #5
    [1,	202,202,202,202,202,202,202,202,202,202,202,202,202,202,202,1,	203,203,203,203,203,203,203,203,203,203,203,203,203,203,1], #6
    [1,	31,	202,202,202,202,202,202,202,202,202,202,202,202,202,202,1,	32,	203,203,203,203,203,203,203,203,203,203,203,203,203,1], #7
    [1,	31,	31,	202,202,202,202,202,202,202,202,202,202,202,13,	202,1,	32,	32,	203,203,203,203,203,203,203,203,203,203,13,	203,1], #8
    [1,	31,	31,	31,	202,202,202,202,202,202,202,202,202,202,202,202,1,	32,	32,	32,	203,203,203,203,203,203,203,203,203,203,203,1], #9
    [1,	31,	31,	31,	31,	202,202,202,202,202,202,202,202,202,202,202,1,	32,	32,	32,	32,	203,203,203,203,203,203,203,203,203,32,1],  #10
    [1,	31,	31,	31,	31,	31,	202,202,202,202,202,41,	41,	41,	202,202,1,	32,	32,	32,	32,	32,	203,203,203,42,	42,	42,	203,203,32,1],  #11
    [1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	201,201,201,1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	201,201,201,1,	1,	1,1],   #12
    [1,	3,	201,201,201,201,4,	201,12,	201,201,201,201,201,201,201,46,	46,	201,201,201,201,201,201,201,201,201,201,201,201,201,1], #13
    [1,	201,201,36,	201,201,201,201,201,201,201,201,201,201,201,201,201,46,	201,201,201,201,201,201,201,201,201,201,201,201,201,1], #14
    [1,	201,36,	201,201,201,201,201,201,201,201,201,201,201,201,201,201,1,	1,	1,	1,	201,201,1,	1,	1,	1,	1,	1,	1,	1,1],   #15
    [1,	201,36,	201,201,201,201,201,201,201,201,201,201,201,201,201,201,1,	10,	44,	205,205,205,1,	204,204,204,204,25,	61,	204,1], #16
    [1,	201,201,201,201,201,201,201,201,201,201,201,2,	201,201,201,201,1,	44,	205,205,205,205,1,	204,204,204,204,25,	204,204,1], #17
    [1,	201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,1,	44,	205,205,205,205,1,	204,204,204,204,204,62,	204,1], #18
    [1,	201,201,201,201,201,201,201,201,201,201,201,26,	201,201,26,	201,1,	205,205,205,205,205,1,	204,204,204,204,35,	204,204,1], #19
    [1,	201,201,201,201,201,201,201,201,201,201,201,26,	201,201,26,	201,1,	24,	24,	205,34,	205,1,	204,204,204,204,35,	204,204,1], #20
    [1,	201,201,201,201,201,201,201,201,201,201,201,26,	201,201,26,	201,1,	9,	24,	205,101,34,	1,	204,204,204,204,204,204,204,1], #21
    [1,	201,201,201,201,201,201,201,201,201,201,201,26,	201,201,26,	201,1,	24,	24,	205,34,	205,1,	204,204,204,204,204,63,	204,1], #22
    [1,	201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,1,	1,	1,	1,	1,	1,	1,	204,204,204,204,204,204,204,1], #23
    [1,	201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,1,	45,	45,	204,204,13,	204,204,204,204,204,204,204,204,1], #24
    [1,	201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,1,	204,204,204,204,204,204,204,204,204,204,204,204,204,1], #25
    [1,	201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,204,204,204,204,204,204,204,204,204,204,204,204,204,1], #26
    [1,	201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,204,204,204,204,204,204,204,204,204,204,204,204,204,1], #27
    [1,	201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,201,1,	204,204,204,204,204,204,204,204,204,204,204,204,204,1], #28
    [1,	201,201,201,1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,1]]   #29
    #0  1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30 31





var layout_map=[  //2 for dinning table  //3 for sofa  //4 for desk_1_ //102 for desk_v //5 for bed_down //6 for bed_left //7 for bedroom desk
// 8 for bedroom chair //9 for v_toilet //10 for bathtub //101 for washhand // 11 for server
// 12 for gateway //13 for wall outlet //26,36,46 for living room chair&couch&are   a //21-bed1,22-bed2,23-bed3 // 31,32,33-area in room// 41,42,43//door in room  //24,34,44 bathroom toilet,handwash,area
    //61 stove, 62 sink, 63 refrigerator//25,35,45 kitchen stove,sink,area
    [1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,1],
    [1, 5,  0,  0,  0,  0,  0,  0,  13, 0,  7,  0,  0,  0,  0,  0,  1,  5,  0,  0,  0,  0,  0,  13, 0,  7,  0,  0,  0,  0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 21, 21, 21, 21, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  22, 22, 22, 22, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 31, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  32, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 31, 31, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  13, 0,  1,  32, 32, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  13, 0,1],
    [1, 31, 31, 31, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  32, 32, 32, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 31, 31, 31, 31, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  32, 32, 32, 32, 0,  0,  0,  0,  0,  0,  0,  0,  0,  32,1],
    [1, 31, 31, 31, 31, 31, 0,  0,  0,  0,  0,  41, 41, 41, 0,  0,  1,  32, 32, 32, 32, 32, 0,  0,  0,  42, 42, 42, 0,  0,  32,1],
    [1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  1,  1,  1,1],
    [1, 3,  0,  0,  0,  0,  4,  0,  12, 0,  0,  0,  0,  0,  0,  0,  46, 46, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 0,  0,  36, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  46, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 0,  36, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,1],
    [1, 0,  36, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  10, 44, 0,  0,  0,  1,  0,  0,  0,  0,  25, 61, 0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  1,  44, 0,  0,  0,  0,  1,  0,  0,  0,  0,  25, 0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  44, 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  62, 0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  26, 0,  0,  26, 0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  35, 0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  26, 0,  0,  26, 0,  1,  24, 24, 0,  34, 0,  1,  0,  0,  0,  0,  35, 0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  26, 0,  0,  26, 0,  1,  9,  24, 0,  101,    34, 1,  0,  0,  0,  0,  0,  0,0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  26, 0,  0,  26, 0,  1,  24, 24, 0,  34, 0,  1,  0,  0,  0,  0,  0,  63, 0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  45, 45, 0,  0,  13, 0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,1],
    [1, 0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,1]];
    
def travtree(root,location):
    for i in range(0,len(root.next)):
        if root.next[i].location==location:
            return i
    return -1


def strcmp(a,b):
    if len(a)!=len(b):
        return -1
    for i in range(0,len(b)):
        if a[i]!=b[i]:
            return -1
    return 1

def dist(goal,loc):
    dist=[0,0,0,0]
    for i in range(0,4):
        gate=goal[i]
        distt=[]
        for j in range(0,len(gate)):
            #comma=gate[j].find(',')
            #gx=int(gate[j][:comma])
            #gy=int(gate[j][comma+1:])
            gx=gate[1]
            gy=gate[0]
            distance=pow(gx-loc[0],2)+pow(gy-loc[1],2)
            distt.append(distance)
        dist[i]=min(distt)
    mini=min(dist)
    s_index=dist.index(mini)
    return s_index

def readdchildren(parent):
    parent.addchild()
    if len(parent.children)>0:
        for x in parent.children:
            readdchildren(x)

def findpath(goals,explored,path):
    neighbor=[]
    last=explored[len(explored)-1]
    comma=last.find(',')
    y=int(last[:comma])
    x=int(last[comma+1:])
    up=y+1

    down=y-1
    left=x-1
    right=x+1


def getchild(parent):
    if parent.location not in explored:
        explored.append(parent.location)
    parent.addchild()

def traaddchild(tmp,goal):
    val=tmp.addchild()
    if val<0:
        tmp.end=True
        return
    for i in range(0,len(tmp.children)):
        child=tmp.children[i]
        loc=str(child.location[0])+','+str(child.location[1])
        if loc==goal:
            print("find loc")
            return

def traverslevel(root,level):     #find node without child in level
    node=None
    tmp=root
    i=tmp.level
    childlist=[]
    while i<level:
        clen=len(tmp.children)
        if clen==0:
            return tmp
        else:
            childlist.append(x for x in tmp.children)
            tmp=childlist.pop(0)
            i=tmp.level

    return node

def findpath(node):
    path=[]
    while node.root!=True:
        loc=node.location
        path.insert(0,loc)
        node=node.parent
    return path


def BFS(initstate,goal):

    frontier=[initstate]
    explored=[]

    while len(frontier)>0:
        state=frontier.pop(0)
        explored.append(state.location)
        loc=str(state.location[0])+','+str(state.location[1])
        if loc==goal:
            path=findpath(state)
            print path
            print("success")
            return

        state.addchild()
        for child in state.children:
            if child not in frontier and child.location not in explored:
                frontier.append(child)

    print("failure")


def shortestpath(location):
    goals={}
    for y in range(0,len(layout_map)):
        for x in range(0,len(layout_map[y])):
            if layout_map[y][x]==41 or layout_map[y][x]==31 or layout_map[y][x]==21 or layout_map[y][x]==22 or layout_map[y][x]==32 or layout_map[y][x]==42:
                goal={str(y)+','+str(x):0}
                goals.update(goal)
    comma=location.find(',')
    y=int(location[:comma])
    x=int(location[comma+1:])
    node=[x,y]
    global explored
    explored.append(node)
    root=Tree(node)
    root.root=True
    root.parent=None
    tmp=root
    for goal in goals:
    #goal=goals.popitem()[0]
        BFS(root,goal)










    print root







def main():
    data=open('bayes_clean_layout.txt','r')
    ex=[]
    f=data.readline()
    right=0
    wrong=0
    while(len(f)>0):
        head=f.find('&')
        gg=f[0:head+1]
        tail=f.find('Bedroom1_Sensor_1',head+1)
        temp=f[head+1:tail]
        label=f[tail+len('Bedroom1_Sensor_1')+1:len(f)-2]
        if label == 'true':
            right=right+1
        else:
            wrong=wrong+1
        temp=gg+temp+label
        ex.append(temp)
        f=data.readline()
    #comp1=float(right)/wrong
    #comp2=float(wrong)/right
    #reward=min(comp1,comp2)
    count=[0,0,0,0,0]  #201,202,203,204,205
    r1=['11,12','12,12','13,12']
    r2=['25,12','26,12','27,12']
    kk=['17,26','17,27']
    tl=['21,15','22,15']
    global gate
    gate=[]
    r11l=[12,11]
    r12l=[12,12]
    r13l=[12,13]
    r21l=[12,15]
    r22l=[12,26]
    r23l=[12,27]
    kk1l=[26,17]
    kk2l=[27,17]
    tl1l=[15,21]
    tl2l=[15,22]
    treelist=[]
    gate.append(r11l)
    gate.append(r12l)
    gate.append(r13l)
    gate.append(r21l)
    gate.append(r22l)
    gate.append(r23l)
    gate.append(kk1l)
    gate.append(kk2l)
    gate.append(tl1l)
    gate.append(tl2l)
    '''for x in gate:
        tree=Tree(x)
        tree.root=True
        tree.prop=1
        tree.level=0
        readdchildren(tree)
        treelist.append(tree)'''


    steps_b1=[]
    steps_b2=[]
    steps_k=[]
    trfs=[]
    trfs_b1=[]
    flag=0
    for i in range(0,len(ex)):
        oex=ex[i]
        tex=oex[2:]
        tex=tex.split(";")
        plaz=[]
        trf=0
        tryflist=[]
        for j in range(0,len(tex)):
            if tex[j]=='try finish':
                trf=trf+1
                tryflist.append(tex[j-1])
        trfs.append(trf)
        tryflist=';'.join(tryflist)
        print('inst '+oex[0:2]+tex[len(tex)-1]+' trytimes='+str(trf)+' '+tryflist)

        shortestpath(tex[0])

        for j in range(1,len(tex)):
            if tex[j]!='try finish' and tex[j]!=' true':
                comma=tex[j].find(',')
                if comma<0:
                    break
                m=int(tex[j][0:comma])
                n=int(tex[j][comma+1:])
                maplo=layout_map[n][m]
                loc=[n,m]
                mindist=dist(gate,loc)
                maploo=[maplo,mindist,n,m]
                #if m==16:
                #    print ("m = 16 n= "+str(n))
                if len(plaz)>0:
                    if m!=plaz[len(plaz)-1][3] or n!=plaz[len(plaz)-1][3]:
                        plaz.append(maploo)
                    #    for tree in treelist:
                    #        neargate=tree.traverse([n,m])
                    #        if neargate is True:
                    #            print (tex[j]+" "+str(j)+ "step near gate "+str(tree.location[0])+","+(str(tree.location[1])))
                    if layout_map[n][m]==41:
                        if flag!=i:
                            steps_b1.append(j)
                            flag=i
                            trfs_b1.append(trfs[i])
                    if layout_map[n][m]==42:
                        if flag!=i:
                            steps_b2.append(j)
                            flag=i
                    if layout_map[n][m]==204:
                        if flag!=i:
                            steps_k.append(j)
                            flag=i

                else:
                    plaz.append(maploo)

                if maplo==204 or maplo==205:
                    print ('wrong place'+str(maplo))
            if tex[j]=='try finish':
                plaz.append(tex[j])
        print(plaz)
    print(count)
    print(len(steps_b1))
    print(len(trfs_b1))
    print(steps_b2)
    print(steps_k)
    #plt.hist(steps_b1,bins=20)
    #plt.show()
    #plt.hist(trfs_b1)
    #plt.show()
    plt.plot(steps_b1,trfs_b1,marker='o',linestyle=' ')
    plt.title("Plot of # of trys v.s. # of steps to bedroom1")
    plt.xlabel("# of steps to bedroom1")
    plt.ylabel("# of tries before submission")
    plt.show()
    plt.hist(steps_b1)
    plt.title("distribution of # of steps preceeds to bedroom1")
    plt.show()
    plt.hist(steps_b2)
    plt.show()
    print(trfs)
    plt.hist(trfs)
    plt.title("distribution of # of trys before submission")
    plt.show()


main()

print(countree)
