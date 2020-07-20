from pglib.general import checked_get_single
import dash_table
import dash_html_components as html


def get_train_args(reader):
    # Read data from database
    try:
        table = reader.get_data_table('train_args')
    except KeyError:
        return []
    launches = reader.execute(table.select().order_by(table.c.id))

    ref_content = launches[0].content

    # Convert to dash table
    children = []
    is_ref = True
    for settings in launches:

        children.append(
            html.Details([
                html.Summary(f'Launch {settings.id}'),
                dash_table.DataTable(
                    id=f'launch-table-{settings.id}',
                    columns=[{"name": i, "id": i}
                             for i in ['Setting', 'Value']],
                    style_cell_conditional=[
                        {
                            'if': {'column_id': 'Value'},
                            'textAlign': 'left'
                        }
                    ],
                    data=[{'Setting': setting, 'Value': str(value)}
                          for setting, value in settings.content.items() if is_ref or value != ref_content[setting]],
                )
            ])
        )
        is_ref = False

    return children
