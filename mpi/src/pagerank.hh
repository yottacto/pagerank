#pragma once
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>
#include <cmath>
#include <mpi.h>
#include "util.hh"
#include "timer.hh"

namespace ice
{

struct node
{
    bool owned{};
    bool boundary{};
    int edge_count{};
    double pr{};
};

struct pagerank
{
    pagerank(std::string const& path, double eps = 1e-6, double damping = 0.85)
        : path(path), eps(eps), damping(damping)
    {
        if (!MPI::Is_initialized())
            MPI::Init();
        rank = MPI::COMM_WORLD.Get_rank();
        size = MPI::COMM_WORLD.Get_size();

        read_partition(path);
        read_graph(path);

        recv_count = 2 * max_boundary_node_count;
        print("recv_count=",            recv_count,            ", ");
        print("max_boundary_node_count=", max_boundary_node_count, ", ");
        recv_buf.reserve(recv_count * size);
        // statistic
        elapsed.resize(size * 2);
    }

    ~pagerank()
    {
        if (!MPI::Is_finalized())
            MPI::Finalize();
    }

    void update_pr()
    {
        recv_buf.resize(recv_count * size, -1);

        std::swap_ranges(
            std::begin(recv_buf),
            std::next(std::begin(recv_buf), recv_count),
            std::next(std::begin(recv_buf), recv_count * rank)
        );

        compute_timer.stop();
        comm_timer.restart();

        MPI::COMM_WORLD.Allgather(
            MPI::IN_PLACE, 0, MPI::DATATYPE_NULL,
            recv_buf.data(), recv_count, MPI::DOUBLE
        );

        for (auto i = 0u; i < recv_buf.size(); i += 2) {
            auto u     = recv_buf[i];
            auto value = recv_buf[i + 1];
            if (u == -1)
                continue;
            pr[u] = value;
        }
        for (auto u : nodes)
            pr[u] = info[u].pr;
    }

    void compute_core()
    {
        recv_buf.clear();
        for (auto u : nodes) {
            auto sum = 0.;
            for (auto v : g[u])
                sum += pr[v] / info[v].edge_count;
            info[u].pr = damping * sum + (1. - damping) / n;
            if (info[u].boundary) {
                recv_buf.emplace_back(u);
                recv_buf.emplace_back(info[u].pr);
            }
            if (std::fabs(info[u].pr - pr[u]) > eps) {
                updated_count++;
            } else {
                converge_count++;
            }
        }

        update_pr();
    }

    template <bool Enabled = false>
    void compute()
    {
        print<Enabled>("computing.\n");

        // statisitc
        updated.clear();
        total_compute = total_comm = 0.;

        total_timer.restart();
        pr.clear();
        pr.resize(n, 1. / n);

        iter = 0;
        for (; converge_count < n; iter++) {
            print<Enabled>("iterating on ", iter, ", \n");
            print<Enabled>("pr[0] ", pr[0], ", \n");

            updated_count = 0;
            converge_count = 0;

            total_timer.start();
            compute_timer.restart();

            compute_core();

            comm_timer.stop();
            total_timer.stop();

            // statistic
            updated.emplace_back(size);
            updated[iter][rank] = updated_count;

            MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &converge_count, 1, MPI::INT, MPI::SUM);
            print<Enabled>("converge count ", converge_count, ", \n");

            auto comm_elapsed = comm_timer.elapsed_seconds();
            auto compute_elapsed = compute_timer.elapsed_seconds();
            // total_comm += comm_elapsed;
            // total_compute += compute_elapsed;

            if (Enabled) {
                elapsed[0] = compute_elapsed;
                elapsed[1] = comm_elapsed;
                if (!rank)
                    MPI::COMM_WORLD.Gather(
                        MPI::IN_PLACE, 0, MPI::DATATYPE_NULL,
                        elapsed.data(), 2, MPI::DOUBLE,
                        0
                    );
                else
                    MPI::COMM_WORLD.Gather(
                        elapsed.data(), 2, MPI::DOUBLE,
                        nullptr, 0, MPI::DATATYPE_NULL,
                        0
                    );

                if (!rank) {
                    auto max_comp = 0.0;
                    auto max_comm = 0.0;
                    for (auto i = 0; i < size; i++) {
                        std::cerr << "rank: " << i
                            << ", compute " << std::setw(5) << elapsed[2 * i]
                            << ", comm " << std::setw(5) << elapsed[2 * i + 1]
                            << std::endl;
                        max_comp = std::max(max_comp, elapsed[2 * i]);
                        max_comm = std::max(max_comm, elapsed[2 * i + 1] + elapsed[2 * i]);
                    }
                    if (max_comm < max_comp)
                        special_round = iter;
                    total_comm += std::max(max_comm - max_comp, 0.);
                    total_compute += max_comp;
                    std::cerr << "rank: a"
                        << ", compute " << std::setw(5) << max_comp
                        << ", comm " << std::setw(5) << max_comm
                        << std::endl;
                }
            }

            print<Enabled>("\n");
        }

        total = total_timer.elapsed_seconds();
        // MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &total_compute, 1, MPI::DOUBLE, MPI::MAX);
        // MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &total_comm,    1, MPI::DOUBLE, MPI::MAX);
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &total,         1, MPI::DOUBLE, MPI::MAX);
        print<Enabled>("total compute elapsed ", total_compute, ", ");
        print<Enabled>("total comm elapsed ", total_comm, "\n");
        print<Enabled>("total time elapsed ", total, "\n");
    }

    std::string binary_file_name(std::string const& path)
    {
        return path + ".pagerank.binary." + std::to_string(size)
            + ".rank." + std::to_string(rank);
    }

    void read_partition(std::string const& base_path)
    {
        auto t{timer{}};
        t.restart();

        auto path = base_path + ".metis.part." + std::to_string(size);
        auto has_binary = std::ifstream{
            binary_file_name(path),
            std::ios::binary
        }.good();
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &has_binary, 1, MPI::BOOL, MPI::LAND);

        if (has_binary) {
            print("reading binary graph partition file.\n");
            auto fin = std::ifstream{binary_file_name(path), std::ios::binary};
            int count;
            bin_read(fin, &count);
            nodes.reserve(count);
            for (int u; count--; ) {
                bin_read(fin, &u);
                nodes.emplace_back(u);
            }
        } else {
            print("reading normal graph partition file.\n");
            auto fin = std::ifstream{path};
            auto fout = std::ofstream{binary_file_name(path), std::ios::binary};
            for (int u = 0, part; fin >> part; u++)
                if (part == rank)
                    nodes.emplace_back(u);
            int count = nodes.size();
            bin_write(fout, &count);
            for (auto u : nodes)
                bin_write(fout, &u);
        }

        t.stop();
        print("read partition elapsed ", t.elapsed_seconds(), "s\n");
    }

    void read_graph(std::string const& path)
    {
        auto t{timer{}};
        t.restart();
        auto has_binary = std::ifstream{
            binary_file_name(path),
            std::ios::binary
        }.good();
        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &has_binary, 1, MPI::BOOL, MPI::LAND);

        if (has_binary) {
            print("reading binary graph file\n");
            auto fin = std::ifstream{binary_file_name(path), std::ios::binary};
            bin_read(fin, &n);
            bin_read(fin, &m);
            g.resize(n);
            info.resize(n);
            for (auto u : nodes)
                info[u].owned = true;
            print("n=", n, ", ");
            print("m=", m, "\n");
            for (auto u = 0; u < n; u++)
                bin_read(fin, &info[u].edge_count);
            int node_count;
            bin_read(fin, &node_count);
            for (int u, size; node_count--; ) {
                bin_read(fin, &u);
                bin_read(fin, &size);
                g[u].reserve(size);
                for (int v; size--; ) {
                    bin_read(fin, &v);
                    g[u].emplace_back(v);
                }
            }

            int boundary_node_count;
            bin_read(fin, &boundary_node_count);
            boundary_nodes.reserve(boundary_node_count);
            for (int u; boundary_node_count--; ) {
                bin_read(fin, &u);
                boundary_nodes.emplace_back(u);
            }
            bin_read(fin, &max_boundary_node_count);
        } else {
            print("reading normal graph file\n");
            auto fin = std::ifstream{path};
            auto fout = std::ofstream{binary_file_name(path), std::ios::binary};
            for (char ch; fin >> ch; ) {
                std::string buf;
                if (ch == '#') {
                    std::getline(fin, buf);
                } else if (ch == 'p') {
                    fin >> buf >> n >> m;
                    bin_write(fout, &n);
                    bin_write(fout, &m);
                    g.resize(n);
                    info.resize(n);
                    for (auto u : nodes)
                        info[u].owned = true;
                } else {
                    int u, v;
                    fin >> u >> v;
                    u--; v--;
                    info[u].edge_count++;
                    if (info[v].owned) {
                        g[v].emplace_back(u);
                        if (!info[u].owned) {
                            boundary_nodes.emplace_back(v);
                        }
                    }
                }
            }

            for (auto u = 0; u < n; u++)
                bin_write(fout, &info[u].edge_count);

            auto node_count = 0;
            for (auto u = 0; u < n; u++)
                if (!g[u].empty())
                    node_count++;
            bin_write(fout, &node_count);
            for (auto u = 0; u < n; u++) {
                if (g[u].empty())
                    continue;
                bin_write(fout, &u);
                int size = g[u].size();
                bin_write(fout, &size);
                for (auto v : g[u])
                    bin_write(fout, &v);
            }
            max_boundary_node_count = boundary_nodes.size();
            MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, &max_boundary_node_count, 1, MPI::INT, MPI::MAX);

            int boundary_node_count = boundary_nodes.size();
            bin_write(fout, &boundary_node_count);
            for (auto u : boundary_nodes)
                bin_write(fout, &u);
            bin_write(fout, &max_boundary_node_count);
        }

        pr.resize(n, 1. / n);
        for (auto u : boundary_nodes)
            info[u].boundary = true;

        MPI::COMM_WORLD.Barrier();

        t.stop();
        print("read graph elapsed ", t.elapsed_seconds(), "s\n");
    }

    template <bool Enabled = true, class T, class U = std::string, class V = std::string>
    void print(T const& x, U const& y = std::string{}, V const& z = std::string{}, bool all = false)
    {
        ::ice::print<Enabled>(rank, x, y, z, all);
    }

    void summary(bool all = false)
    {
        if (all || !rank) {
            std::cout << "n=" << n << " m=" << m << "\n";
        }
    }

    void print_statistic()
    {
        if (!rank)
            std::cerr << "\nupdaed per iter\n";
        for (auto i = 0; i < iter; i++) {
            MPI::COMM_WORLD.Allgather(
                MPI::IN_PLACE, 0, MPI::DATATYPE_NULL,
                updated[i].data(), 1, MPI::INT
            );
            if (!rank) {
                std::cerr << "iter " << i << ": ";
                for (auto u : updated[i])
                    std::cerr << std::setw(7) << u << " ";
                std::cerr << "\n";
            }
        }

        MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, pr.data(), n, MPI::INT, MPI::MIN);
        if (!rank) {
            auto sum = 0.;
            for (auto i = 0; i < n; i++)
                sum += pr[i];
            std::cerr << "\nall pr value avg: " << sum / n << "\n\n";

            std::cerr << "speical round: " << special_round << "\n";
            std::cerr << "| " << size << " | " << iter - 1 << " | " << total << " | " << total_comm << " | " << total_compute << "\n";
        }
    }

    // graph data path
    std::string path;
    int rank;
    int size;

    // total number of nodes [n] and edges [m]
    int n;
    int m;
    std::vector<std::vector<int>> g;
    std::vector<node> info;

    // nodes belong to this rank
    std::vector<int> nodes;
    std::vector<int> boundary_nodes;
    int max_boundary_node_count;

    int iter;
    std::vector<double> pr;
    double eps;
    double damping;

    int converge_count;
    int recv_count;
    std::vector<double> recv_buf;

    // statistic
    int special_round{-1};
    std::vector<std::vector<int>> updated;
    double total;
    double total_compute;
    double total_comm;
    timer compute_timer;
    timer comm_timer;
    timer total_timer;
    int updated_count;
    std::vector<double> elapsed;
};

} // namespace ice

