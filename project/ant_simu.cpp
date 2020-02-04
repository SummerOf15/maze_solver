#include <vector>
#include <iostream>
#include <random>
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
# include "mpi.h"
# include "omp.h"
# include <time.h>

using namespace std;

void advance_time( const labyrinthe& land, pheromone& phen, 
                   const position_t& pos_nest, const position_t& pos_food,
                   std::vector<ant>& ants, std::size_t& cpteur )
{
    vector<position_t> new_pos;
    new_pos.resize(ants.size());
#pragma omp parallel for num_threads(3)
    for ( size_t i = 0; i < ants.size(); ++i )
    {
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
        new_pos[i]=ants[i].get_position();
    }
    MPI_Send(&new_pos[0], sizeof(position_t)*ants.size(),MPI_BYTE, 0, 13, MPI_COMM_WORLD);
}


int main(int nargs, char* argv[])
{

    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    cout<<world_rank<<endl;
    size_t food_quantity = 0;
    const dimension_t dims{48,48};// Dimension du labyrinthe
    
    // Location du nid
    position_t pos_nest{dims.first/2,dims.second/2};
    // Location de la nourriture
    position_t pos_food{dims.first-1,dims.second-1};
                          
    const std::size_t life = int(dims.first*dims.second);
    const int nb_ants = 2*dims.first*dims.second; // Nombre de fourmis
    const double eps = 0.75;  // Coefficient d'exploration
    const double alpha=0.97; // Coefficient de chaos
    const double beta=0.999; // Coefficient d'évaporation
                                 // 
    // Définition du coefficient d'exploration de toutes les fourmis.
    ant::set_exploration_coef(eps);
    
    std::vector<ant> ants;

    MPI_Status status;
    ants.reserve(nb_ants);
    for ( size_t i = 0; i < nb_ants; ++i )
        ants.emplace_back(pos_nest, life);

    labyrinthe laby(dims);
    pheromone phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);
    if(world_rank==0)
    {    
        clock_t start=clock();

        // synchrnize labyrinthe infomation
        int l=0;
        MPI_Recv(&l, 1,MPI_INT, 1, 9, MPI_COMM_WORLD,&status);
        cout<<"length"<<l<<endl;
        vector<unsigned char> labydata;
        labydata.resize(l);
        MPI_Recv(&labydata[0], l,MPI_UNSIGNED_CHAR, 1, 10, MPI_COMM_WORLD,&status);
        laby.set_laby_data(labydata);

        pheromone phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);
        gui::context graphic_context(nargs, argv);
        gui::window& win =  graphic_context.new_window(h_scal*laby.dimensions().second,h_scal*laby.dimensions().first+266);

        // new calculated position for movement
        vector<position_t> new_pos;
        new_pos.resize(ants.size());

        display_t displayer( laby, phen, pos_nest, pos_food, ants, win );
        gui::event_manager manager;
       
        manager.on_key_event(int('q'), [] (int code) { exit(0); });
        manager.on_display([&] { displayer.display(food_quantity); win.blit(); });

        // time counter
        int count=0;

        manager.on_idle([&] () 
        { 
        	// receive new position
            MPI_Recv(&new_pos[0], sizeof(position_t)*ants.size(),MPI_BYTE, 1, 13, MPI_COMM_WORLD,&status);

            for ( size_t i = 0; i < new_pos.size(); ++i )
            {
                ants[i].push_new_pos(phen, laby, pos_food, pos_nest, food_quantity, new_pos[i]);
            }
            phen.do_evaporation();
            phen.update();
            count+=1;
            if (count==300)
            {
                clock_t end=clock();
                cout<<"duration:"<<(end - start) / CLOCKS_PER_SEC<<endl;

            }
            displayer.display(food_quantity); 
            win.blit(); 
        });
        manager.loop();
      
        
    }
    else if(world_rank==1)
    {
    	// send labyrinthe information to synchrosize with processus 0
        int l=laby.get_laby_data().size();
        MPI_Send(&l,1,MPI_INT,0,9,MPI_COMM_WORLD);// data size
        MPI_Send(&laby.get_laby_data()[0],l,MPI_UNSIGNED_CHAR,0,10,MPI_COMM_WORLD);
        
        while(1)
        {
            advance_time(laby, phen, pos_nest, pos_food, ants, food_quantity);     
        }
    }
    
    MPI_Finalize();
    return 0;
}