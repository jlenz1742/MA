import math


def define_activated_region(graph, coords_sphere, r_sphere):

    activated_edge_ids = []

    for edge in range(graph.ecount()):

        # Target or source point inside sphere -------------------------------------------------------------------------

        # p1 = graph.es[edge].source
        # x_1 = graph.vs[p1]['x_coordinate'] / math.pow(10, 6)
        # y_1 = graph.vs[p1]['y_coordinate'] / math.pow(10, 6)
        # z_1 = graph.vs[p1]['z_coordinate'] / math.pow(10, 6)
        #
        # p2 = graph.es[edge].target
        # x_2 = graph.vs[p2]['x_coordinate'] / math.pow(10, 6)
        # y_2 = graph.vs[p2]['y_coordinate'] / math.pow(10, 6)
        # z_2 = graph.vs[p2]['z_coordinate'] / math.pow(10, 6)
        #
        # x_3 = coords_sphere['x'] / math.pow(10, 6)
        # y_3 = coords_sphere['y'] / math.pow(10, 6)
        # z_3 = coords_sphere['z'] / math.pow(10, 6)
        #
        # radius = r_sphere / math.pow(10, 6)
        #
        # distance_p1 = math.sqrt(math.pow(x_1 - x_3, 2) + math.pow(y_1 - y_3, 2) + math.pow(z_1 - z_3, 2))
        # distance_p2 = math.sqrt(math.pow(x_2 - x_3, 2) + math.pow(y_2 - y_3, 2) + math.pow(z_2 - z_3, 2))
        #
        # if distance_p1 < radius:
        #
        #     activated_edge_ids.append(edge)
        #
        # elif distance_p2 < radius:
        #
        #     activated_edge_ids.append(edge)
        #
        # else:
        #
        #     continue

        # Part of line segment inside sphere (or at least tangential) --------------------------------------------------

        p1 = graph.es[edge].source
        x_1 = graph.vs[p1]['x_coordinate'] / math.pow(10, 6)
        y_1 = graph.vs[p1]['y_coordinate'] / math.pow(10, 6)
        z_1 = graph.vs[p1]['z_coordinate'] / math.pow(10, 6)

        p2 = graph.es[edge].target
        x_2 = graph.vs[p2]['x_coordinate'] / math.pow(10, 6)
        y_2 = graph.vs[p2]['y_coordinate'] / math.pow(10, 6)
        z_2 = graph.vs[p2]['z_coordinate'] / math.pow(10, 6)

        x_3 = coords_sphere['x'] / math.pow(10, 6)
        y_3 = coords_sphere['y'] / math.pow(10, 6)
        z_3 = coords_sphere['z'] / math.pow(10, 6)

        radius = r_sphere / math.pow(10, 6)

        a = math.pow(x_2 - x_1, 2) + math.pow(y_2 - y_1, 2) + math.pow(z_2 - z_1, 2)
        b = 2 * ((x_2 - x_1)*(x_1 - x_3) + (y_2 - y_1)*(y_1 - y_3) + (z_2 - z_1)*(z_1 - z_3))
        c = math.pow(x_3, 2) + math.pow(y_3, 2) + math.pow(z_3, 2) + math.pow(x_1, 2) + math.pow(y_1, 2) + math.pow(z_1, 2) - 2 * (x_3 * x_1 + y_3 * y_1 + z_3 * z_1) - math.pow(radius, 2)

        value = math.pow(b, 2) - 4 * a * c

        if value >= 0:

            u_1 = (-b + math.sqrt(value)) / (2 * a)
            u_2 = (-b - math.sqrt(value)) / (2 * a)

            # Line segment doesnt intersect but is inside sphere

            if u_1 < 0 and u_2 > 1:

                activated_edge_ids.append(edge)

            elif u_2 < 0 and u_1 > 1:

                activated_edge_ids.append(edge)

            # Line segment intersects at one point

            elif (0 <= u_1 <= 1) and (u_2 > 1 or u_2 < 0):

                activated_edge_ids.append(edge)

            elif (0 <= u_2 <= 1) and (u_1 > 1 or u_1 < 0):

                activated_edge_ids.append(edge)

            # Line segment intersects at two points

            elif (0 <= u_2 <= 1) and (0 <= u_1 <= 1):

                activated_edge_ids.append(edge)

            # Line segment is tangential

            elif (u_1 == u_2) and (0 <= u_1 <= 1):

                activated_edge_ids.append(edge)

            else:

                continue

        else:

            continue

    for activated_edge in activated_edge_ids:

        graph.es[activated_edge]['Activated'] = 1

    return graph


def modifiable_region(graph, coords_sphere, r_sphere):

    activated_edge_ids = []

    name = 'modi_radius_' + str(r_sphere)

    for edge in range(graph.ecount()):

        # Target or source point inside sphere -------------------------------------------------------------------------

        # p1 = graph.es[edge].source
        # x_1 = graph.vs[p1]['x_coordinate'] / math.pow(10, 6)
        # y_1 = graph.vs[p1]['y_coordinate'] / math.pow(10, 6)
        # z_1 = graph.vs[p1]['z_coordinate'] / math.pow(10, 6)
        #
        # p2 = graph.es[edge].target
        # x_2 = graph.vs[p2]['x_coordinate'] / math.pow(10, 6)
        # y_2 = graph.vs[p2]['y_coordinate'] / math.pow(10, 6)
        # z_2 = graph.vs[p2]['z_coordinate'] / math.pow(10, 6)
        #
        # x_3 = coords_sphere['x'] / math.pow(10, 6)
        # y_3 = coords_sphere['y'] / math.pow(10, 6)
        # z_3 = coords_sphere['z'] / math.pow(10, 6)
        #
        # radius = r_sphere / math.pow(10, 6)
        #
        # distance_p1 = math.sqrt(math.pow(x_1 - x_3, 2) + math.pow(y_1 - y_3, 2) + math.pow(z_1 - z_3, 2))
        # distance_p2 = math.sqrt(math.pow(x_2 - x_3, 2) + math.pow(y_2 - y_3, 2) + math.pow(z_2 - z_3, 2))
        #
        # if distance_p1 < radius:
        #
        #     activated_edge_ids.append(edge)
        #
        # elif distance_p2 < radius:
        #
        #     activated_edge_ids.append(edge)
        #
        # else:
        #
        #     continue

        # Part of line segment inside sphere (or at least tangential) --------------------------------------------------

        p1 = graph.es[edge].source
        x_1 = graph.vs[p1]['x_coordinate'] / math.pow(10, 6)
        y_1 = graph.vs[p1]['y_coordinate'] / math.pow(10, 6)
        z_1 = graph.vs[p1]['z_coordinate'] / math.pow(10, 6)

        p2 = graph.es[edge].target
        x_2 = graph.vs[p2]['x_coordinate'] / math.pow(10, 6)
        y_2 = graph.vs[p2]['y_coordinate'] / math.pow(10, 6)
        z_2 = graph.vs[p2]['z_coordinate'] / math.pow(10, 6)

        x_3 = coords_sphere['x'] / math.pow(10, 6)
        y_3 = coords_sphere['y'] / math.pow(10, 6)
        z_3 = coords_sphere['z'] / math.pow(10, 6)

        radius = r_sphere / math.pow(10, 6)

        a = math.pow(x_2 - x_1, 2) + math.pow(y_2 - y_1, 2) + math.pow(z_2 - z_1, 2)
        b = 2 * ((x_2 - x_1)*(x_1 - x_3) + (y_2 - y_1)*(y_1 - y_3) + (z_2 - z_1)*(z_1 - z_3))
        c = math.pow(x_3, 2) + math.pow(y_3, 2) + math.pow(z_3, 2) + math.pow(x_1, 2) + math.pow(y_1, 2) + math.pow(z_1, 2) - 2 * (x_3 * x_1 + y_3 * y_1 + z_3 * z_1) - math.pow(radius, 2)

        value = math.pow(b, 2) - 4 * a * c

        if value >= 0:

            u_1 = (-b + math.sqrt(value)) / (2 * a)
            u_2 = (-b - math.sqrt(value)) / (2 * a)

            # Line segment doesnt intersect but is inside sphere

            if u_1 < 0 and u_2 > 1:

                activated_edge_ids.append(edge)

            elif u_2 < 0 and u_1 > 1:

                activated_edge_ids.append(edge)

            # Line segment intersects at one point

            elif (0 <= u_1 <= 1) and (u_2 > 1 or u_2 < 0):

                activated_edge_ids.append(edge)

            elif (0 <= u_2 <= 1) and (u_1 > 1 or u_1 < 0):

                activated_edge_ids.append(edge)

            # Line segment intersects at two points

            elif (0 <= u_2 <= 1) and (0 <= u_1 <= 1):

                activated_edge_ids.append(edge)

            # Line segment is tangential

            elif (u_1 == u_2) and (0 <= u_1 <= 1):

                activated_edge_ids.append(edge)

            else:

                continue

        else:

            continue

    for activated_edge in activated_edge_ids:

            graph.es[activated_edge][name] = 1

    return graph
