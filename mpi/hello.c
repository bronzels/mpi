#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main (int argc, char **argv)
{
  int myid, numprocs,source;
  MPI_Status status;
  char message[100];
  MPI_Init (&argc, &argv);  /* starts MPI */

  MPI_Comm_rank (MPI_COMM_WORLD, &myid);    /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &numprocs);    /* get number of processes */
  
    if(myid!=0)
    {
        strcpy(message,"Hello World!");
        MPI_Send(message,strlen(message)+1,MPI_CHAR,0,99,MPI_COMM_WORLD);
    }
    else
    {
        for (source=1;source<numprocs;source++)
        {
            MPI_Recv(message,100,MPI_CHAR,source,99,MPI_COMM_WORLD,&status);
            printf("I am Process %d. I recv string '%s' from process %d.\n",myid,message,source);
        }
    }

MPI_Finalize();

  return 0;
}
