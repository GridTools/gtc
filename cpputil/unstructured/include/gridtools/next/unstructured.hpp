#pragma once

// TODO namespace?
namespace gridtools {
    namespace next {
        namespace topology {
            struct vertex {};
            struct edge {};
            struct cell {};
        } // namespace topology
        using topology::cell;
        using topology::edge;
        using topology::vertex;

        struct neighbor {};

        namespace dim {
            struct k {};
        } // namespace dim
    }     // namespace next
} // namespace gridtools
