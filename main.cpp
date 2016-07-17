#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stack>
#include <queue>
#include <iostream>
#include <stack>
#include <mpi.h>
#include <vector>
#include <climits>
#include <omp.h>

using namespace std;

typedef struct
{
    int *cities;
    int num_cities;
    int cost;
} Tour;

#define City_count(tour) (tour->num_cities)
#define Tour_cost(tour) (tour->cost)
#define Last_city(tour) (tour->cities[(tour->num_cities)-1])
#define Tour_city(tour,i) (tour->cities[(i)])
#define Cost_c(length, i, city) (length+Cost[i][city])

int comm_sz; // número de nós de processamento
queue<Tour*> local_queue;
int n, E;
int minPath = INT_MAX;
int **adj = NULL;  // Matriz de Adjacentes
int **Cost = NULL; // Matriz de Custos
int my_rank;
int best_global;
#define TOUR 2

int Cost_Calculate(int *path, int num_cities);
void visit(const int city, int hops, int length, int *path);
void process_tree(int *bufrecv, int size, int cities);
inline bool visited(Tour *tour, int city);
Tour* addCity(Tour *tour, int newcity);
bool IsBestCost(const int &Curr_tourCost);
int Create_Queues();
inline void Free(Tour *tour);

inline void Free(Tour *tour)
{
    if(tour != NULL)
    {
        if(tour->cities != NULL)
        {
            free(tour->cities);
        }
        free(tour);
    }
}

int main(int argc, char *argv[])
{
    int bufrecv[10000];
    int quotient, remainder;
    int a, b, d;
    double start, finish;
    int required=MPI_THREAD_SERIALIZED;
    int provided;
    
    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    if (provided < required)
    {
        // Insufficient support, degrade to 1 thread and warn the user
        if (my_rank == 0)
        {
            cout << "Warning: This MPI implementation provides insufficient" << " threading support." << endl;
        }   omp_set_num_threads(1);
    }
    
    
    if(my_rank ==0)
    {
        cin >> n >> E;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    adj = new int*[n];
    Cost = new int*[n];
    for(int i = 0; i < n; i++)
    {
        adj[i] = new int[n];
        Cost[i] = new int[n];
        
        memset(adj[i], 0, n*sizeof(int));
        memset(Cost[i], 0, n*sizeof(int));
    }
    
    if(my_rank == 0)
    {
        for(int i = 0; i < E; i++)
        {
            cin >> a >> b >> d;
            Cost[a][b] = d;
            Cost[b][a] = d;
            adj[a][b] = 1;
            adj[b][a] = 1;
        }
    }
    
    for(int i = 0; i < n; i++)
    MPI_Bcast(&(Cost[i][0]), n, MPI_INT, 0, MPI_COMM_WORLD);
    for(int i = 0; i < n; i++)
    MPI_Bcast(&(adj[i][0]), n, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(my_rank == 0)
    {
        int queue_size = Create_Queues();
        
        vector<int> tours;
        
        while(!local_queue.empty())
        {
            Tour *temp = local_queue.front();
            local_queue.pop();
            
            for(int j=0;j<n+1;j++)
            {
                tours.push_back(temp->cities[j]);
            }
            
            Free(temp);
        }
        
        if(comm_sz>1)
        {
            quotient = queue_size/(comm_sz-1);
            remainder = queue_size%(comm_sz-1);
            
            int next = 0;
            int send = 0;
            
            for(int i = 0; i < comm_sz-1; i++)
            {
                send = (quotient)*(n+1);
                MPI_Send(&send, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(tours.data() + next, send, MPI_INT, i, 0, MPI_COMM_WORLD);
                next += (quotient)*(n+1);
            }
            
            send = (remainder)*(n+1);
            MPI_Send(&send, 1, MPI_INT, comm_sz-1, 0, MPI_COMM_WORLD);
            MPI_Send(tours.data() + next, send, MPI_INT, comm_sz-1, 0, MPI_COMM_WORLD);
        }
        else
        {
            start = MPI_Wtime();
            process_tree(tours.data(), 1, n);
            IsBestCost(minPath);
            finish = MPI_Wtime();
            cout << "NProc: " << comm_sz << endl;
            cout << minPath << endl;
            cout << "Tempo: " << finish-start << endl;
            cout << "-------------" << endl;
            MPI_Finalize();
            return 0;
        }
    }
    MPI_Status status;
    int size;
    
    MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(bufrecv, size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    process_tree(bufrecv, size, n);
    IsBestCost(minPath);
    finish = MPI_Wtime();
    
    struct
    {
        int cost;
        int rank;
    } loc_data, global_data;
    
    loc_data.cost = minPath;
    loc_data.rank = my_rank;
    
    /*for(int i=0;i<comm_sz;i++)
    {
    if(my_rank == i) cout << "rank : " << my_rank << "-" << minPath << endl;
    }*/
    
    MPI_Allreduce(&loc_data, &global_data, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
    
    if(my_rank == 0)
    {
        cout << "NProc: " << comm_sz << endl;
        cout << global_data.cost << endl;
        cout << "Tempo: " << finish-start << endl;
        cout << "-----------" << endl;
    }
    
    MPI_Finalize();
    return 0;
}

bool IsBestCost(const int &Curr_tourCost)
{
    int cost;
    int check;
    MPI_Status status;
    
    while(true)
    {
        #pragma omp critical
        {
            MPI_Iprobe(MPI_ANY_SOURCE, TOUR, MPI_COMM_WORLD, &check, &status);
        }
        
        if (check)
        {
            #pragma omp critical
            {
                MPI_Recv(&cost, 1, MPI_INT, status.MPI_SOURCE, TOUR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            if (cost < minPath)
            {
                #pragma omp critical (newMin)
                {
                    if (cost < minPath) minPath = cost;
                }
            }
        }
        else break;
        }
    
    bool sendBsend = false;
    
    if(Curr_tourCost < minPath)
    {
        #pragma omp critical (newMin)
        {
            if(Curr_tourCost < minPath)
            {
                minPath = Curr_tourCost;
                sendBsend= true;
            }
        }
        if(sendBsend)
        {
            for(int k=0;k<comm_sz;k++)
            {
                if(k != my_rank)
                {
                    #pragma omp critical
                    {
                        MPI_Bsend(&(minPath), 1, MPI_INT, k, TOUR, MPI_COMM_WORLD);
                    }
                }
            }
        }
    }
}


inline int visited (int city, int hops, const int *path)
{
    int i = 0;
    for (; i < hops-3; i+=4)
    {
        if (path[i] == city) return 1;
        if (path[i+1] == city) return 1;
        if (path[i+2] == city) return 1;
        if (path[i+3] == city) return 1;
    }
    for (; i < hops; ++i) if (path[i] == city) return 1;
        
    return 0;
}

void visit(const int city, int hops, int length, int *path)
{
    path[hops-1] = city;
    
    if (length >= minPath)
    {
        return;
    }
    else if(hops == n)
    {
        if(adj[city][0])
        {
            IsBestCost(Cost_c(length,city,0));
        }
        return;
    }
    
    for (int i = 0; i < n; i++)
    {
        if(!visited(i, hops, path) && adj[city][i])
        {
            visit(i, hops+1, Cost_c(length,i,city), path);
        }
    }
    
}

void process_tree(int *bufrecv, int size, int cities)
{
    int j = 0, i = 0;
    int count = 0;
    int path[50][50];
    int temp[50];
    
    memset( &path[0][0], -1, sizeof(path) );
    
    /*for(int l=0;l<size;l++)
    {
    cout << bufrecv[l] << "*";
    }
    cout << endl;*/
    
    while(j < size)
    {
        while(j < size && bufrecv[j] != -1)
        {
            path[count][i++] = bufrecv[j++];
        }
        
        temp[count] = i;
        count++;
        
        while(j < size && bufrecv[j] == -1)
        {
            j++;
        }
        i = 0;
    }
    
    
    int last, temp_cost;
    
    #pragma omp parallel for private(last,temp_cost)
    for(int k = 0; k < count; k++)
    {
        //for(int i=0;i<temp[k];i++) cout << path[k][i] << "*";
        //cout << endl;
        last = path[k][temp[k]-1];
        temp_cost = Cost_Calculate(path[k], temp[k]);
        
        //cout << "Cost " << temp_cost << endl;
        //cout << endl;
        
        visit(last,temp[k],temp_cost,path[k]);
    }
}


inline int Cost_Calculate(int *path, int num_cities)
{
    int cost = 0;
    
    for(int i = 0; i < num_cities - 1; i++)
    {
        cost += Cost[path[i]][path[i+1]];
        //cout << Cost[path[i]][path[i+1]] << "From: " << path[i] << " TO: " << path[i+1] << endl;
    }
    
    return cost;
}

void InitTour(Tour **tour)
{
    *tour = new Tour[1];
    (*(tour))->cities = new int[sizeof(int)*(n+1)];
    (*(tour))->cities[0] = 0;
    (*(tour))->num_cities = 1;
    (*(tour))->cost = 0;
    
    for(int i = 1; i < n+1; i++)
    {
        (*(tour))->cities[i] = -1;
    }
}

Tour* addCity(Tour *tour, int newcity)
{
    Tour **newTour = new Tour*[1];
    *newTour = new Tour[1];
    (*(newTour))->cities = new int[sizeof(int)*(n+1)];
    
    for(int i = 0; i < n+1; i++)
    {
        (*newTour)->cities[i] = Tour_city(tour,i);
    }
    
    (*newTour)->num_cities = tour->num_cities;
    
    (*newTour)->cities[(*newTour)->num_cities] = newcity;
    
    (*newTour)->num_cities++;
    
    return *newTour;
}

inline bool visited(Tour *tour, int city)
{
    for(int i = 0; i < City_count(tour); i++)
    {
        if(Tour_city(tour,i) == city) return true;
    }
    
    return false;
}

int Create_Queues()
{
    Tour **tour = new Tour*[1];
    
    InitTour(tour);
    
    local_queue.push(*tour);
    int num = 1;
    int NThreads;
    
    #pragma omp parallel
    {
        NThreads = omp_get_num_threads();
    }
    
    while(num < comm_sz+NThreads && !local_queue.empty())
    {
        Tour *temp = local_queue.front();
        local_queue.pop();
        num --;
        
        for(int i = 1; i < n; i++)
        {
            if(adj[temp->cities[City_count(temp) - 1]][i] && !visited(temp, i))
            {
                local_queue.push(addCity(temp, i));
                num++;
            }
        }
        
        if(temp != NULL)
        {
            if(temp->cities != NULL)
            {
                free(temp->cities);
            }
            free(temp);
        }
    }
    
    return num;
}
