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
            System.out.println("To Vertex " + i + " is " + (distances[i] == Integer.MAX_VALUE ? "Infinity" : distances[i]));
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
        } while (updated);

        printSolution(distances, startVertex);
    }

    public static void main(String[] args) {
        int vertices = 5;
        int edges = 8;
        Graph graph = new Graph(vertices, edges);

        graph.addEdge(0, 0, 1, -1);
        graph.addEdge(1, 0, 2, 4);
        graph.addEdge(2, 1, 2, 3);
        graph.addEdge(3, 1, 3, 2);
        graph.addEdge(4, 1, 4, 2);
        graph.addEdge(5, 3, 2, 5);
        graph.addEdge(6, 3, 1, 1);
        graph.addEdge(7, 4, 3, -3);

        graph.bellmanFord(0);

        int[][] adjacencyMatrix = {
                {0, -1, 4, Integer.MAX_VALUE, Integer.MAX_VALUE},
                {Integer.MAX_VALUE, 0, 3, 2, 2},
                {Integer.MAX_VALUE, Integer.MAX_VALUE, 0, Integer.MAX_VALUE, Integer.MAX_VALUE},
                {Integer.MAX_VALUE, 1, 5, 0, Integer.MAX_VALUE},
                {Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE, -3, 0}
        };

        System.out.println("\nDistance Vector Routing:");
        graph.distanceVectorRouting(adjacencyMatrix, 0);
    }
}
