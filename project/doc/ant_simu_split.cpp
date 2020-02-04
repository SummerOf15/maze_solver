#include <vector>
#include <iostream>
#include <random>
#include <stdio.h>
#include "labyrinthe.hpp"
#include "ant.hpp"
#include "pheromone.hpp"
# include "gui/context.hpp"
# include "gui/colors.hpp"
# include "gui/point.hpp"
# include "gui/segment.hpp"
# include "gui/triangle.hpp"
# include "gui/quad.hpp"
# include "gui/event_manager.hpp"
# include "display.hpp"
#include "mpi.h"
using namespace std;

# define send_data_tag 1
# define receive_data_tag 2



void decompose_domain(int domain_size, int world_rank, int world_size, int* subdomain_start, int* subdomain_size)
{
    if (world_size > domain_size)
    {
        // Assume the domain size is greater than the world size
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    *subdomain_start = domain_size / world_size * world_rank;
    *subdomain_size = domain_size / world_size;
    if (world_rank == world_size - 1)
    {
        // give remainders to the last subdomain
        *subdomain_size += domain_size % world_size;
    }
}

void send_outgoing_ants(std::vector<ant>* upgoing_walkers, std::vector<ant>* downgoing_walkers, int world_rank, int world_size)
{
    // send the data as an array of MPI_BYTES to the next process
    // the last process send the data to process 0
    if(upgoing_walkers->size()>0)
    {
        MPI_Send(upgoing_walkers->data(), upgoing_walkers->size() * sizeof(ant), MPI_BYTE, (world_rank - 1) % world_size, 0, MPI_COMM_WORLD);
        printf("Process %d sending %d outgoing walkers to process %d\n", world_rank, upgoing_walkers->size(), (world_rank - 1) % world_size);
        upgoing_walkers->clear();
    }
    if (downgoing_walkers->size()>0)
    {
        MPI_Send(downgoing_walkers->data(), downgoing_walkers->size() * sizeof(ant), MPI_BYTE, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
        printf("Process %d sending %d outgoing walkers to process %d\n", world_rank, downgoing_walkers->size(), (world_rank + 1) % world_size);
        downgoing_walkers->clear();
    }
    // clear the outgoing walkers list
    
    
}

void receive_incoming_walkers(std::vector<ant>* incoming_walkers, int world_rank, int world_size)
{
    MPI_Status status;
    // receive from the data from bottom direction.
    // if current process is 0, then receive from the last process

    int incoming_walkers_size;
    int incoming_rank = world_rank+1;

    if (incoming_rank<world_size)
    {
        MPI_Probe(incoming_rank, 0, MPI_COMM_WORLD, &status);
        // allocate buffer for incoming message
        
        MPI_Get_count(&status, MPI_BYTE, &incoming_walkers_size);
        // retrive the first n walkers
        incoming_walkers->resize(incoming_walkers_size / sizeof(ant));
        MPI_Recv(incoming_walkers->data(), incoming_walkers_size, MPI_BYTE, incoming_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // receive data from top direction
    std::vector<ant> downcoming_walkers;
    incoming_rank = world_rank-1;
    if (incoming_rank>=0)
    {
        MPI_Probe(incoming_rank, 0, MPI_COMM_WORLD, &status);
        // allocate buffer for incoming message
        MPI_Get_count(&status, MPI_BYTE, &incoming_walkers_size);
        if(incoming_walkers_size>0)
        {
            // retrive the first n walkers
            downcoming_walkers.resize(incoming_walkers_size / sizeof(ant));
            MPI_Recv(downcoming_walkers.data(), incoming_walkers_size, MPI_BYTE, incoming_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int i=0;i<incoming_walkers_size;i++)
            {
                incoming_walkers->push_back(downcoming_walkers[i]);
            }
            //incoming_walkers->insert(incoming_walkers->end(),downcoming_walkers->begin(),downcoming_walkers->end());
        }
    }
    
}

void advance_time( const labyrinthe& land, pheromone& phen, const position_t& pos_nest, const position_t& pos_food,
                   std::vector<ant>& ants, std::size_t& cpteur,int subdomain_start,int subdomain_size, int domain_size, 
                    std::vector<ant>* upgoing_ants, std::vector<ant>* downgoing_ants, int rank, int num_procs)
{
    for (size_t i = 0; i < ants.size(); ++i)
    {
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
        position_t last_pos = ants[i].get_position();
        if (last_pos.first < subdomain_start)
        {
            upgoing_ants->push_back(ants[i]);
        }
        else if (last_pos.first >= subdomain_start + subdomain_size)
        {
            downgoing_ants->push_back(ants[i]);
        }
    }
    //printf("%d+%d",upgoing_ants->size(),downgoing_ants->size());
    // send_outgoing_ants(upgoing_ants, downgoing_ants, rank, num_procs);
    phen.do_evaporation();
    phen.update();
}
int main(int nargs, char* argv[])
{
    int rank,num_procs;
    MPI_Init(&nargs,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const dimension_t dims{ 64,64 };// Dimension du labyrinthe
    const std::size_t life = int(dims.first * dims.second);
    const int nb_ants = 2 * dims.first * dims.second; // Nombre de fourmis
    const double eps = 0.75;  // Coefficient d'exploration
    const double alpha = 0.97; // Coefficient de chaos

    const double beta = 0.999; // Coefficient d'évaporation
                        
    labyrinthe laby(dims);
    // Location du nid
    position_t pos_nest{dims.first/2,dims.second/2};
    // Location de la nourriture
    position_t pos_food{dims.first-1,dims.second-1};
                          
    int subdomain_start, subdomain_size;
    std::vector<ant> upgoing_ants, downgoing_ants;
    // Définition du coefficient d'exploration de toutes les fourmis.
    ant::set_exploration_coef(eps);
    std::vector<ant> ants;
    if(rank == 1)
    {
        cout<<sizeof(ant)<<endl;
        // On va créer toutes les fourmis dans le nid :
        ants.reserve(nb_ants);
        for ( size_t i = 0; i < nb_ants; ++i )
            ants.emplace_back(pos_nest, life);
        // On crée toutes les fourmis dans la fourmilière.
        pheromone phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);

        gui::context graphic_context(nargs, argv);
        gui::window& win =  graphic_context.new_window(h_scal*laby.dimensions().second,h_scal*laby.dimensions().first+266);
        display_t displayer( laby, phen, pos_nest, pos_food, ants, win );
        size_t food_quantity = 0;

        gui::event_manager manager;
        manager.on_key_event(int('q'), [](int code) { exit(0); });
        manager.on_display([&] { displayer.display(food_quantity); win.blit(); });


        decompose_domain(dims.first, rank, num_procs, &subdomain_start, &subdomain_size);
        std::cout << "totally divided" << num_procs << " partitions" << std::endl;

        manager.on_idle([&]() {
            advance_time(laby, phen, pos_nest, pos_food, ants, food_quantity, subdomain_start, subdomain_size, dims.first,
                &upgoing_ants, &downgoing_ants, rank, num_procs);
            displayer.display(food_quantity);
            win.blit();
            });
        std::cout<<upgoing_ants.size()<<std::endl;
        manager.loop();
    }
    else if (rank % 2 == 0)
    {
        std::cout<<"rank "<<rank<<std::endl;

        cout<<ants.size();
        //send all outgoing walkers to the next process
        send_outgoing_ants(&upgoing_ants, &downgoing_ants, rank, num_procs);
        //receiving all the new incoming walkers
        receive_incoming_walkers(&ants, rank, num_procs);
    }
    else
    {
        std::cout<<"rank "<<rank<<std::endl;
       
        //receiving all the new incoming walkers
        receive_incoming_walkers(&ants, rank, num_procs);
        //send all outgoing walkers to the next process
        send_outgoing_ants(&upgoing_ants, &downgoing_ants, rank, num_procs);

        
    }
    printf("Process %d received %d incoming walkers\n", rank, ants.size());


    MPI_Finalize();
    return 0;
}