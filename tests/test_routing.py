import pytest
from optimization.graph import build_grid_graph
from optimization.problem import Stop, Vehicle
from optimization.solver import solve

def test_feasible_route_is_independently_valid():
    g=build_grid_graph(); result=solve(g,"n0_0",[Stop("b1","n1_1",10),Stop("b2","n2_2",15)],[Vehicle("t1",100)])
    assert result.status=="FEASIBLE"; assert result.validation["valid"]; assert result.routes[0].road_path[0]==result.routes[0].road_path[-1]=="n0_0"
    assert result.solver == "ortools_cvrp_guided_local_search"

def test_capacity_infeasibility_is_explicit():
    g=build_grid_graph(); result=solve(g,"n0_0",[Stop("b1","n1_1",200,"4" if False else 4)],[Vehicle("t1",100)])
    assert result.status=="INFEASIBLE"; assert result.unserved[0]["reason"]=="demand_exceeds_any_vehicle_capacity"

def test_empty_manifest_is_valid_empty_solution():
    result=solve(build_grid_graph(),"n0_0",[],[Vehicle("t1",100)])
    assert result.status=="FEASIBLE" and result.validation["valid"] and result.routes[0].road_path==[]

def test_invalid_inputs_fail():
    with pytest.raises(ValueError): Stop("x","n0_0",-1)
    with pytest.raises(ValueError): Vehicle("x",-1)
    with pytest.raises(ValueError): solve(build_grid_graph(),"n0_0",[Stop("x","n1_1",1),Stop("x","n2_2",1)],[Vehicle("t",3)])
