#include<stdio.h>
# include <stdlib.h>
# include <math.h>
#include <string.h>

double Exhaustive_Search(double a, double b);
double objective_function(double x);

main()
{	
	double a, b; /*Ranges of x*/
	printf("Enter\n a = lower limit of x\n b = upper limit on x\n");
	scanf("%lf %lf",&a,&b);
	
	Exhaustive_Search(a,b);
	/* Bounding_Phase(); */	
}

double objective_function(double x)
{
	return(x*x + (54/x));
}

double Exhaustive_Search(double a, double b)
{
	int i, n, feval;
	double delta;
	double x1, x2, x3, f1, f2, f3;
	FILE *out;
	
	printf("\n**********************************\n");
	printf("Exhaustive Search Method\n");
	printf("Enter 'n' = number of steps\t");
	scanf("%d",&n);
	/*Step 1*/
	delta = (b-a)/(double) n; /*Set delta value*/
	x1 = a; x2 = x1 + delta; x3 = x2 + delta; /*New points*/
	feval = 0; /*function evaluation*/
	f1 = objective_function(x1); /*Calculate objective_function*/
	f2 = objective_function(x2);
	f3 = objective_function(x3);
	i = 1; /*Number of iterations*/
	feval = feval + 3;
	out = fopen("Exhaustive_Search_iterations.out","w");/*Output file*/
	fprintf(out,"#It\tx1\tx2\tx3\tf(x1)\tf(x2)\tf(x3)\n");
	do{
		fprintf(out,"%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\n",i,x1,x2,x3,f1,f2,f3);
		/*Step 2: Termination condition */
		if(f1 >= f2){
			if(f2 <= f3){
				/* break; */
			}
		}
		/*If not terminated, update values*/
		x1 = x2; x2 = x3; x3 = x2 + delta;
		f1 = f2;
		f2 = f3;
		f3 = objective_function(x3);
		feval++;
		i = i + 1;
	}while(x3 <= b);/*Step 3*/
	printf("\n**********************************\n");
	printf("The minimum point lies between (%lf,%lf)",x1,x3);
	printf("\n#Total number of function evaluations: %d",feval);
	/*Store in the file*/
	fprintf(out,"\n#The minimum point lies between (%lf,%lf)",x1,x3);
	fprintf(out,"\n#Total number of function evaluations: %d",feval);
	fclose(out);
	return(1);
}
