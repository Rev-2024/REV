import java.util.Arrays; 

class Graph {
 class Edge {
 int source, destination, weight;
 
 Edge(int source, int destination, int weight) {
 this.source = source;
 this.destination = destination;
 this.weight = weight;
 }
 }
 int vertices, edges; 
 Edge[] edgeList;
 
 Graph(int vertices, int edges) {
 this.vertices = vertices;
 this.edges = edges;
 edgeList = new Edge[edges]; 
 }
 void addEdge(int edgeIndex, int source, int destination, int weight) {
    edgeList[edgeIndex] = new Edge(source, destination, weight); 
 }
 void bellmanFord(int startVertex) {
 int[] distances = new int[vertices];
 Arrays.fill(distances, Integer.MAX_VALUE);
 distances[startVertex] = 0; 
 for (int i = 1; i < vertices; i++) {
 for (int j = 0; j < edges; j++) {
 int u = edgeList[j].source; 
 int v = edgeList[j].destination; 
 int weight = edgeList[j].weight; 
 
 if (distances[u] != Integer.MAX_VALUE && distances[u] + weight < distances[v]) {
 distances[v] = distances[u] + weight;
 }
 }
 }

 for (int j = 0; j < edges; j++) {
 int u = edgeList[j].source; 
 int v = edgeList[j].destination; 
 int weight = edgeList[j].weight; 

 if (distances[u] != Integer.MAX_VALUE && distances[u] + weight < distances[v]) { 
    System.out.println("Graph contains a negative-weight cycle");
 return; 
 }
 }
 
 printSolution(distances, startVertex);
 }
 
 void printSolution(int[] distances, int startVertex) {
 System.out.println("Vertex distances from source vertex " + startVertex + ":");
 for (int i = 0; i < vertices; i++) {
 System.out.println("To Vertex " + i + " is " + distances[i]);
 }
 }
 
 void distanceVectorRouting(int[][] graph, int startVertex) {
 int[] distances = new int[vertices];
 Arrays.fill(distances, Integer.MAX_VALUE); 
 distances[startVertex] = 0; 
 boolean updated; 
 
 do {
 updated = false;
 
 for (int u = 0; u < vertices; u++) {
 for (int v = 0; v < vertices; v++) {
    if (graph[u][v] != Integer.MAX_VALUE && distances[u] != Integer.MAX_VALUE &&
 distances[u] + graph[u][v] < distances[v]) {
 distances[v] = distances[u] + graph[u][v];
 updated = true; 
 }
 }
 }
 }
 }
 } while (updated); 
 