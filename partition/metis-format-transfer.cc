// ml:ccf += -g
// ml:opt = 0
#include <algorithm>
#include <fstream>
#include <limits>
#include <vector>
#include <set>
#include <map>
#include <utility>

#include <iostream>


struct graph
{
    graph(std::string const& path) : path(path)
    {
        read_graph(path);
        transfer(path + ".metis");
    }

    template <class T>
    auto inrange(T x, T l, T r)
    {
        return !(x < l) && x < r;
    }

    auto get_id(int u)
    {
        return std::distance(
            std::begin(id),
            std::lower_bound(std::begin(id), std::end(id), u)
        ) + 1;
    }

    void preprocessing(std::string const& path)
    {
        auto fin = std::ifstream{path};
        for (char ch; fin >> ch; ) {
            std::string buf;
            if (ch == '#') {
                std::getline(fin, buf);
                continue;
            }
            if (ch == 'p') {
                fin >> n >> m;
                id.reserve(m * 2);
                continue;
            }
            int u, v;
            fin >> u >> v;
            id.emplace_back(u);
            id.emplace_back(v);
        }
        std::sort(std::begin(id), std::end(id));
        id.erase(std::unique(std::begin(id), std::end(id)), std::end(id));
    }

    // read original data and map the id into [1, n], write to path+".new"
    void read_graph(std::string const& path)
    {
        preprocessing(path);
        int min = 1<<30;
        int max = -1;
        auto fin = std::ifstream{path};
        auto fout = std::ofstream{path + ".new"};
        for (char ch; fin >> ch; ) {
            std::string buf;
            if (ch == '#') {
                std::getline(fin, buf);
                continue;
            }
            if (ch == 'p') {
                fin >> n >> m;
                std::cerr << n << " " << m << "\n";
                fout << "p sp " << n << " " << m << "\n";
                g.resize(n + 1);
                continue;
            }
            int u, v;
            fin >> u >> v;
            u = get_id(u);
            v = get_id(v);
            fout << "a " << u << " " << v << "\n";
            min = std::min(min, std::min(u, v));
            max = std::max(max, std::max(u, v));
            g[u].emplace(v);
            g[v].emplace(u);
        }
        m = 0;
        for (auto i = 1; i <= n; i++)
            m += g[i].size();
        m /= 2;
        std::cerr << id.size() << " " << min << " " << max << "\n";
    }

    void transfer(std::string const& path)
    {
        auto fout = std::ofstream{path};
        fout << n << " " << m << " 0\n";
        for (auto i = 1; i <= n; i++) {
            auto first = true;
            for (auto const& v : g[i]) {
                if (!first)
                    fout << " ";
                else
                    first = false;
                fout << v;
            }
            fout << "\n";
        }
    }

    // graph data path
    std::string path;
    // total number of nodes [n] and edges [m]
    int n;
    int m;
    // for node <from>, store {<to> -> edge}, because graph may contain
    // duplicate edges
    std::vector<std::set<int>> g;

    std::vector<int> id;
};

int main()
{
    graph g("../dataset/web-Google.txt");
}

